import os
import argparse
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from sklearn.metrics import confusion_matrix, f1_score
from tensorflow.keras.callbacks import TensorBoard
import datetime

from dataset.HAR_dataset import HARDataset
from utils.logger import initialize_logger, record_result
from configs import args, dict_to_markdown
from tqdm import tqdm, trange

def set_seed(seed):
    # fix seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_model(n_feat, n_cls):
    if args.seed:
        set_seed(args.seed)

    if args.model == 'BaselineCNN':
        from models.Baseline_CNN import Baseline_CNN
        model = Baseline_CNN(n_feat, n_cls, 128)
    elif args.model == 'DeepConvLSTM':
        from models.DeepConvLSTM import DeepConvLSTM
        model = DeepConvLSTM(n_feat, n_cls)
    elif args.model == 'LSTMConvNet':
        from models.LSTM_CNN import LSTMConvNet
        model = LSTMConvNet(n_feat, n_cls)
    elif args.model == 'EarlyFusion':
        from models.CAGE import CAGE_EarlyFusion
        model = CAGE_EarlyFusion(n_feat, n_cls)
    else:
        raise NotImplementedError(f"{args.model} is not implemented")

    if args.load_model != '':
        model.load_weights(args.load_model)

    # Learning rate schedule
    initial_learning_rate = args.learning_rate
    decay_steps = 20
    decay_rate = 0.5
    learning_rate_fn = PiecewiseConstantDecay(
        boundaries=[decay_steps * i for i in range(1, args.epochs // decay_steps)],
        values=[initial_learning_rate * (decay_rate ** i) for i in range(args.epochs // decay_steps + 1)]
    )

    optimizer = Adam(learning_rate=learning_rate_fn, weight_decay=args.weight_decay)
    return model, optimizer

@tf.function
def train_step(model, x, y, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return predictions, loss

def train():
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_set.data, train_set.labels)
    ).shuffle(10000).batch(args.batch_size)
    
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (val_set.data, val_set.labels)
    ).batch(args.batch_size)

    model, optimizer = get_model(n_feat, n_cls)
    
    loss_fn = SparseCategoricalCrossentropy(from_logits=True)
    train_accuracy = SparseCategoricalAccuracy()
    val_accuracy = SparseCategoricalAccuracy()
    
    best_f1 = 0.0
    best_epoch = 0

    # Logging setup
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.save_folder, 'logs', current_time)
    summary_writer = tf.summary.create_file_writer(log_dir)
    result = open(os.path.join(args.save_folder, 'result'), 'w+')

    print("==> training...")
    for epoch in trange(args.epochs, desc='Training_epoch'):
        # Training
        train_loss = 0
        train_predictions = []
        train_labels = []
        
        for x, y in train_dataset:
            predictions, loss = train_step(model, x, y, optimizer, loss_fn)
            train_loss += loss
            train_accuracy(y, predictions)
            train_predictions.extend(tf.argmax(predictions, axis=1).numpy())
            train_labels.extend(y.numpy())

        train_f1 = f1_score(train_labels, train_predictions, average='weighted')
        
        # Validation
        val_loss = 0
        val_predictions = []
        val_labels = []
        
        for x, y in val_dataset:
            predictions = model(x, training=False)
            val_loss += loss_fn(y, predictions)
            val_accuracy(y, predictions)
            val_predictions.extend(tf.argmax(predictions, axis=1).numpy())
            val_labels.extend(y.numpy())

        val_f1 = f1_score(val_labels, val_predictions, average='weighted')

        # Logging
        with summary_writer.as_default():
            tf.summary.scalar('train_accuracy', train_accuracy.result(), step=epoch)
            tf.summary.scalar('train_f1', train_f1, step=epoch)
            tf.summary.scalar('train_loss', train_loss, step=epoch)
            tf.summary.scalar('val_accuracy', val_accuracy.result(), step=epoch)
            tf.summary.scalar('val_f1', val_f1, step=epoch)
            tf.summary.scalar('val_loss', val_loss, step=epoch)

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_accuracy.result() * 100
            best_epoch = epoch
            model.save_weights(os.path.join(args.save_folder, 'best'))
            c_mat = confusion_matrix(val_labels, val_predictions)

        # Reset metrics
        train_accuracy.reset_states()
        val_accuracy.reset_states()

    # Save final model
    model.save_weights(os.path.join(args.save_folder, 'final'))
    
    print('Done')
    print(f'Best performance at epoch {best_epoch}, accuracy: {best_acc:.2f}%, F1: {best_f1:.4f}')
    print(c_mat)
    record_result(result, best_epoch, best_acc, best_f1, c_mat)

def evaluate(model, dataset, epoch, is_test=True, mode='best'):
    if is_test:
        model.load_weights(os.path.join(args.save_folder, mode))
    
    accuracy = SparseCategoricalAccuracy()
    loss_fn = SparseCategoricalCrossentropy(from_logits=True)
    
    test_loss = 0
    predictions = []
    labels = []
    
    for x, y in dataset:
        pred = model(x, training=False)
        test_loss += loss_fn(y, pred)
        accuracy(y, pred)
        predictions.extend(tf.argmax(pred, axis=1).numpy())
        labels.extend(y.numpy())

    acc_test = accuracy.result() * 100
    f1_test = f1_score(labels, predictions, average='weighted')

    if is_test:
        print(f'=> test acc: {acc_test:.2f}%, test F1: {f1_test:.4f}')
        logger.info(f'=> test acc: {acc_test:.2f}%, test F1: {f1_test:.4f}')
        c_mat = confusion_matrix(labels, predictions)
        result = open(os.path.join(args.save_folder, 'result'), 'a+')
        record_result(result, epoch, acc_test, f1_test, c_mat)
    else:
        logger.info(f'=> val acc: {acc_test:.2f}%, val F1: {f1_test:.4f}')
        logger.info(f'=> loss: {test_loss:.4f}')

    return acc_test, f1_test

if __name__ == "__main__":
    print(dict_to_markdown(vars(args)))

    # Dataset preparation
    train_set = HARDataset(dataset=args.dataset, split='train', 
                          window_width=args.window_width, clean=args.no_clean, 
                          include_null=args.no_null, use_portion=args.train_portion)
    val_set = HARDataset(dataset=args.dataset, split='val',
                        window_width=args.window_width, clean=args.no_clean,
                        include_null=args.no_null)
    test_set = HARDataset(dataset=args.dataset, split='test',
                         window_width=args.window_width, clean=args.no_clean,
                         include_null=args.no_null)

    if args.normalize:
        train_set.normalize(train_set.mean, train_set.std)
        val_set.normalize(train_set.mean, train_set.std)
        test_set.normalize(train_set.mean, train_set.std)

    n_feat = train_set.feat_dim
    n_cls = train_set.n_actions

    # Create save directory
    args.save_folder = os.path.join(args.model_path, args.dataset, args.model, args.trial)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    # Initialize logger
    log_dir = os.path.join(args.save_folder, 'train.log')
    logger = initialize_logger(log_dir)

    # Training
    train()

    # Testing
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_set.data, test_set.labels)
    ).batch(args.batch_size)
    
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (val_set.data, val_set.labels)
    ).batch(args.batch_size)

    model, _ = get_model(n_feat, n_cls)
    evaluate(model, test_dataset, -1, mode='best')
    evaluate(model, test_dataset, -2, mode='final')
    evaluate(model, val_dataset, -3, mode='final')