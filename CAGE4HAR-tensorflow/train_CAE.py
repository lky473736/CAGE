import os
import argparse
import time
import random
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import confusion_matrix, f1_score
from dataset.HAR_dataset import HARDataset

use_cuda = tf.config.list_physical_devices('GPU')
device = '/GPU:0' if use_cuda else '/CPU:0'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_model(n_feat, n_cls):
    if args.seed:
        set_seed(args.seed)

    from models.ConvAE import CAE, MLP_Classifier
    model = CAE(input_shape=(n_feat,)).to(device)
    classifier = MLP_Classifier(num_classes=n_cls).to(device)

    optimizer_model = optimizers.Adam(learning_rate=args.learning_rate)
    optimizer_classifier = optimizers.Adam(learning_rate=args.learning_rate)
    scheduler_model = tf.keras.optimizers.schedules.StepDecay(args.learning_rate, step_size=25, decay_rate=0.8)
    scheduler_classifier = tf.keras.optimizers.schedules.StepDecay(args.learning_rate, step_size=25, decay_rate=0.8)

    if args.load_model:
        model.load_weights(args.load_model)
    
    return model, classifier, optimizer_model, optimizer_classifier, scheduler_model, scheduler_classifier

def train():
    train_loader = HARDataset(dataset=args.dataset, split='train', window_width=args.window_width, clean=args.no_clean, include_null=args.no_null, use_portion=args.train_portion).get_dataloader(batch_size=args.batch_size)
    val_loader = HARDataset(dataset=args.dataset, split='val', window_width=args.window_width, clean=args.no_clean, include_null=args.no_null).get_dataloader(batch_size=args.batch_size)
    
    model, classifier, optimizer_model, optimizer_classifier, scheduler_model, scheduler_classifier = get_model(n_feat, n_cls)

    n_device = n_feat // 6  # (accel, gyro) * (x, y, z)

    criterion_recon = tf.keras.losses.MeanSquaredError()
    criterion_cls = tf.keras.losses.SparseCategoricalCrossentropy()

    best_f1 = 0.0
    best_epoch = 0

    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(args.save_folder, 'model_best.h5'), save_best_only=True)
    tensorboard_callback = TensorBoard(log_dir=args.save_folder)

    print("==> training autoencoder...")
    for epoch in range(150):
        model.train()
        total_loss = 0
        total_num = 0

        for data in train_loader:
            data = tf.convert_to_tensor(data).to(device)
            with tf.GradientTape() as tape:
                output = model(data)
                loss = criterion_recon(output, data)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer_model.apply_gradients(zip(gradients, model.trainable_variables))

            total_loss += loss.numpy() * len(data)
            total_num += len(data)

        scheduler_model(epoch)
        total_loss /= total_num
        print(f'Epoch: [{epoch}/{args.epochs}] - loss: {total_loss:.4f}')

    print("==> training classifier...")
    model.eval()
    for epoch in range(200):
        classifier.train()
        total_loss = 0
        total_num = 0
        label_list = []
        predicted_list = []

        for data, label in train_loader:
            data, label = tf.convert_to_tensor(data).to(device), tf.convert_to_tensor(label).to(device)
            with tf.GradientTape() as tape:
                _, hidden = model(data)
                output = classifier(hidden)
                loss = criterion_cls(output, label)
            gradients = tape.gradient(loss, classifier.trainable_variables)
            optimizer_classifier.apply_gradients(zip(gradients, classifier.trainable_variables))

            total_loss += loss.numpy() * len(label)
            predicted = tf.argmax(output, axis=1)
            label_list.append(label)
            predicted_list.append(predicted)
            total_num += len(label)

        scheduler_classifier(epoch)
        total_loss /= total_num
        label = tf.concat(label_list, axis=0).numpy()
        predicted = tf.concat(predicted_list, axis=0).numpy()
        acc_train = (predicted == label).sum() * 100.0 / total_num
        f1_train = f1_score(label, predicted, average='weighted')
        print(f'Epoch: [{epoch}/{args.epochs}] - loss: {total_loss:.4f}, train acc: {acc_train:.2f}%, train F1: {f1_train:.4f}')

        val_acc, val_f1 = evaluate(model, classifier, val_loader, epoch)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch

    print(f'Done. Best performance achieved at epoch {best_epoch}, best F1: {best_f1:.4f}')

def evaluate(model, classifier, eval_loader, epoch, mode='best'):
    model.eval()
    classifier.eval()
    total_loss = 0
    total_num = 0

    criterion_cls = tf.keras.losses.SparseCategoricalCrossentropy()

    label_list = []
    predicted_list = []
    for data, label in eval_loader:
        data, label = tf.convert_to_tensor(data).to(device), tf.convert_to_tensor(label).to(device)
        with tf.GradientTape() as tape:
            _, hidden = model(data)
            output = classifier(hidden)
            loss = criterion_cls(output, label)
        total_loss += loss.numpy() * len(label)
        predicted = tf.argmax(output, axis=1)
        label_list.append(label)
        predicted_list.append(predicted)
        total_num += len(label)

    total_loss /= total_num
    label = tf.concat(label_list, axis=0).numpy()
    predicted = tf.concat(predicted_list, axis=0).numpy()
    acc_test = (predicted == label).sum() * 100.0 / total_num
    f1_test = f1_score(label, predicted, average='weighted')

    print(f'=> {mode} acc: {acc_test:.2f}%, {mode} F1: {f1_test:.4f}')
    return acc_test, f1_test

if __name__ == "__main__":
    # Assuming args are parsed from a configuration file or command line arguments
    train()    
    test_loader = HARDataset(dataset=args.dataset, split='test', window_width=args.window_width, clean=args.no_clean, include_null=args.no_null).get_dataloader(batch_size=args.batch_size)
    model, classifier, _, _, _, _ = get_model(n_feat, n_cls)
    evaluate(model, classifier, test_loader, -1, mode='best')
    evaluate(model, classifier, test_loader, -2, mode='final')
