import os
import argparse
import time
import random
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from sklearn.metrics import confusion_matrix, f1_score
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np

from dataset.HAR_dataset import HARDataset
from utils.logger import initialize_logger, record_result
from configs import args, dict_to_markdown

from models import CAGE

use_cuda = tf.test.is_gpu_available() # if cuda is used at my computer
device = '/GPU:0' if use_cuda else '/CPU:0' # cuda is here ? use GPU : use CPU

def set_seed(seed):
    # fix seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_model(n_feat, n_cls):
    if args.seed:
        set_seed(args.seed)

    if args.lambda_ssl > 0:
        proj_dim = args.proj_dim
    else:
        proj_dim = 0
    
    model = CAGE.CAGE(n_feat // 2, n_cls, proj_dim)
    if args.load_model != '':
        model.load_weights(args.load_model, by_name=True)
    
    optimizer = optimizers.Adam(learning_rate=args.learning_rate, 
                                weight_decay=args.weight_decay)
    scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=args.learning_rate,
        decay_steps=20,
        decay_rate=0.5
    )
    return model, optimizer, scheduler

def train() :
    train_dataset = HARDataset(dataset=args.dataset, split='train', window_width=args.window_width, clean=args.no_clean, include_null=args.no_null, use_portion=args.train_portion)
    
    val_dataset = HARDataset(dataset=args.dataset, split='val', window_width=args.window_width, clean=args.no_clean, include_null=args.no_null)
    
    model, optimizer, scheduler = get_model(n_feat, n_cls)

    n_device = n_feat // 6

    criterion_cls = losses.CategoricalCrossentropy()
    best_f1 = 0.0
    best_epoch = 0

    # logging
    log_dir = os.path.join(args.save_folder, 'train.log')
    logger = initialize_logger(log_dir)
    tensorboard_callback = TensorBoard(log_dir=args.save_folder)

    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(args.save_folder, 'best.weights.h5'), 
                                          save_best_only=True, 
                                          save_weights_only=True)

    print("==> training...")
    for epoch in range(args.epochs):
        model.train()
        total_num, total_loss = 0, 0
        ssl_gt, ssl_pred, cls_gt, cls_pred = [], [], [], []
        for idx, (data, label) in enumerate(train_dataset):
            data, label = tf.convert_to_tensor(data), tf.convert_to_tensor(label)

            accel_x = data[:, :3 * n_device, :]
            gyro_x = data[:, 3 * n_device:, :]
            ssl_output, cls_output = model(accel_x, gyro_x)
            ssl_label = tf.range(tf.shape(ssl_output)[0])

            ssl_loss = (criterion_cls(ssl_output, ssl_label) + criterion_cls(tf.transpose(ssl_output), ssl_label)) / 2
            cls_loss = criterion_cls(cls_output, label)
            loss = ssl_loss * args.lambda_ssl + cls_loss * args.lambda_cls

            with tf.GradientTape() as tape:
                loss = ssl_loss * args.lambda_ssl + cls_loss * args.lambda_cls
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            total_loss += loss.numpy() * len(label)
            ssl_predicted = tf.argmax(ssl_output, axis=1)
            cls_predicted = tf.argmax(cls_output, axis=1)
            ssl_gt.append(ssl_label)
            ssl_pred.append(ssl_predicted)
            cls_gt.append(label)
            cls_pred.append(cls_predicted)
            total_num += len(label)
            if idx % 20 == 0:
                print(f'Epoch: [{epoch}/{args.epochs}], Batch: [{idx+1}] - loss: {loss.numpy():.4f}')

        total_loss /= total_num
        label = tf.concat(cls_gt, axis=0).numpy()
        predicted = tf.concat(cls_pred, axis=0).numpy()
        acc_train = np.sum(predicted == label) * 100.0 / total_num
        f1_train = f1_score(label, predicted, average='weighted')
        label2 = tf.concat(ssl_gt, axis=0).numpy()
        predicted2 = tf.concat(ssl_pred, axis=0).numpy()
        acc_train2 = np.sum(predicted2 == label2) * 100.0 / len(label2)
        logger.info(f'Epoch: [{epoch}/{args.epochs}] - loss: {total_loss:.4f}, train acc: {acc_train:.2f}%, train F1: {f1_train:.4f} | ssl acc: {acc_train2:.2f}%')

        tensorboard_callback.on_epoch_end(epoch, {
            'Train/Accuracy_cls': acc_train,
            'Train/F1_cls': f1_train,
            'Train/Accuracy_ssl': acc_train2,
            'Train/Loss': total_loss
        })

        acc_test, f1_test = evaluate(model, val_dataset, epoch, is_test=False)
        scheduler = scheduler(epoch)
        if epoch % 50 == 0:
            model.save_weights(os.path.join(args.save_folder, f'epoch{epoch}.h5'))

        if f1_test > best_f1:
            best_f1 = f1_test
            best_epoch = epoch
            model.save_weights(os.path.join(args.save_folder, 'best.h5'))

    model.save_weights(os.path.join(args.save_folder, 'final.h5'))
    print(f'Best performance achieved at epoch {best_epoch}, best acc: {acc_test:.2f}%, best F1: {best_f1:.4f}')

    label = tf.concat(cls_gt, axis=0).numpy()
    predicted = tf.concat(cls_pred, axis=0).numpy()
    c_mat = confusion_matrix(label, predicted)
    record_result(args.save_folder + '/result', best_epoch, acc_test, best_f1, c_mat)
    tensorboard_callback.on_epoch_end(epoch, {
        'Test/Accuracy_cls': acc_test,
        'Test/F1_cls': f1_test
    })

def evaluate(model, eval_dataset, epoch, is_test=True):
    model.eval()

    criterion_cls = losses.CategoricalCrossentropy()
    n_device = n_feat // 6

    total_num, ssl_total_loss, cls_total_loss = 0, 0, 0
    ssl_gt, ssl_pred, cls_gt, cls_pred = [], [], [], []
    for idx, (data, label) in enumerate(eval_dataset):
        data, label = tf.convert_to_tensor(data), tf.convert_to_tensor(label)

        accel_x = data[:, :3 * n_device, :]
        gyro_x = data[:, 3 * n_device:, :]
        ssl_output, cls_output = model(accel_x, gyro_x)
        ssl_label = tf.range(tf.shape(ssl_output)[0])

        ssl_loss = (criterion_cls(ssl_output, ssl_label) + criterion_cls(tf.transpose(ssl_output), ssl_label)) / 2
        cls_loss = criterion_cls(cls_output, label)
        ssl_total_loss += ssl_loss.numpy() * len(label)
        cls_total_loss += cls_loss.numpy() * len(label)

        ssl_predicted = tf.argmax(ssl_output, axis=1)
        cls_predicted = tf.argmax(cls_output, axis=1)
        ssl_gt.append(ssl_label)
        ssl_pred.append(ssl_predicted)
        cls_gt.append(label)
        cls_pred.append(cls_predicted)
        total_num += len(label)

    ssl_total_loss /= total_num
    cls_total_loss /= total_num
    label = tf.concat(cls_gt, axis=0).numpy()
    predicted = tf.concat(cls_pred, axis=0).numpy()
    acc_test = np.sum(predicted == label) * 100.0 / total_num
    f1_test = f1_score(label, predicted, average='weighted')
    label2 = tf.concat(ssl_gt, axis=0).numpy()
    predicted2 = tf.concat(ssl_pred, axis=0).numpy()
    acc_test2 = np.sum(predicted2 == label2) * 100.0 / len(label2)

    print(f'Validation acc: {acc_test:.2f}%, Validation F1: {f1_test:.4f} / ssl acc: {acc_test2:.2f}%')

    if is_test:
        c_mat = confusion_matrix(label, predicted)
        record_result(args.save_folder + '/result', epoch, acc_test, f1_test, c_mat)

    return acc_test, f1_test

if __name__ == "__main__":
    print(dict_to_markdown(vars(args)))

    # get set
    train_set = HARDataset(dataset=args.dataset, split='train', window_width=args.window_width, clean=args.no_clean, include_null=args.no_null, use_portion=args.train_portion)
    val_set = HARDataset(dataset=args.dataset, split='val', window_width=args.window_width, clean=args.no_clean, include_null=args.no_null)
    test_set = HARDataset(dataset=args.dataset, split='test', window_width=args.window_width, clean=args.no_clean, include_null=args.no_null)
    if args.normalize:
        train_set.normalize(train_set.mean, train_set.std)
        val_set.normalize(train_set.mean, train_set.std)
        test_set.normalize(train_set.mean, train_set.std)

    n_feat = train_set.feat_dim
    n_cls = train_set.n_actions

    args.save_folder = os.path.join(args.model_path, args.dataset, 'CAGE', args.trial)
    #if not args.no_clean:
    #    args.save_folder = os.path.join(args.model_path, args.dataset + '_Xclean', 'CAGE', args.trial)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    # logging   
    log_dir = os.path.join(args.save_folder, 'train.log')
    logger = initialize_logger(log_dir)
    tensorboard_callback = TensorBoard(log_dir=args.save_folder)

    #train
    train()

    # test
    test_dataset = HARDataset(dataset=args.dataset, split='test', window_width=args.window_width, clean=args.no_clean, include_null=args.no_null)
    val_dataset = HARDataset(dataset=args.dataset, split='val', window_width=args.window_width, clean=args.no_clean, include_null=args.no_null)
    model, _, _ = get_model(n_feat, n_cls)
    evaluate(model, test_dataset, -1, mode='best')
    evaluate(model, test_dataset, -2, mode='final')
    evaluate(model, val_dataset, -3, mode='final')