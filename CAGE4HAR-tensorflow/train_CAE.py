import os
import argparse
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import MSE, SparseCategoricalCrossentropy
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm, trange

# Assuming HARDataset and models are defined similarly as in PyTorch
from dataset.HAR_dataset import HARDataset
from models.ConvAE import CAE, MLP_Classifier

use_cuda = tf.test.is_gpu_available()
device = '/gpu:0' if use_cuda else '/cpu:0'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_model(n_feat, n_cls):
    if args.seed:
        set_seed(args.seed)

    model = CAE(n_feat)
    classifier = MLP_Classifier(n_cls)
    optimizer_model = optimizers.Adam(learning_rate=args.learning_rate)
    optimizer_classifier = optimizers.Adam(learning_rate=args.learning_rate)

    if args.load_model != '':
        model.load_weights(args.load_model)

    return model, classifier, optimizer_model, optimizer_classifier

def train():
    train_set = HARDataset(dataset=args.dataset, split='train', window_width=args.window_width)
    val_set = HARDataset(dataset=args.dataset, split='val', window_width=args.window_width)

    n_feat = train_set.feat_dim
    n_cls = train_set.n_actions

    model, classifier, optimizer_model, optimizer_classifier = get_model(n_feat, n_cls)

    loss_recon = MSE()
    loss_cls = SparseCategoricalCrossentropy()

    best_f1 = 0.0
    best_epoch = 0

    # Training Autoencoder
    for epoch in trange(150, desc='Training_epoch'):
        model.train()
        total_loss = 0
        for data in train_set:
            data = tf.convert_to_tensor(data, dtype=tf.float32)
            with tf.GradientTape() as tape:
                output = model(data)
                loss = loss_recon(output, data)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer_model.apply_gradients(zip(gradients, model.trainable_variables))

            total_loss += loss.numpy()

        print(f'Epoch: {epoch}, Loss: {total_loss}')

    print("==> Training classifier...")
    for epoch in trange(200, desc='Training_epoch'):
        classifier.train()
        total_loss = 0
        label_list = []
        predicted_list = []
        for data, label in train_set:
            data = tf.convert_to_tensor(data, dtype=tf.float32)
            label = tf.convert_to_tensor(label, dtype=tf.int64)
            with tf.GradientTape() as tape:
                encoded = model(data)
                output = classifier(encoded)
                loss = loss_cls(output, label)
            gradients = tape.gradient(loss, classifier.trainable_variables)
            optimizer_classifier.apply_gradients(zip(gradients, classifier.trainable_variables))

            total_loss += loss.numpy()
            predicted = tf.argmax(output, axis=1)
            label_list.extend(label.numpy())
            predicted_list.extend(predicted.numpy())

        f1_train = f1_score(label_list, predicted_list, average='weighted')

        if f1_train > best_f1:
            best_f1 = f1_train
            best_epoch = epoch
            model.save_weights(os.path.join(args.save_folder, 'model_best.h5'))
            classifier.save_weights(os.path.join(args.save_folder, 'classifier_best.h5'))

        print(f'Epoch: {epoch}, F1: {f1_train}')

    print(f'Best performance achieved at epoch {best_epoch}, best F1: {best_f1}')

def evaluate(model, eval_loader, epoch, is_test=True, mode='best'):
    model.eval()
    total_loss = 0
    label_list = []
    predicted_list = []

    for data, label in eval_loader:
        data = tf.convert_to_tensor(data, dtype=tf.float32)
        label = tf.convert_to_tensor(label, dtype=tf.int64)
        output = model(data)
        loss = loss_cls(output, label)

        total_loss += loss.numpy()
        predicted = tf.argmax(output, axis=1)
        label_list.extend(label.numpy())
        predicted_list.extend(predicted.numpy())

    f1_test = f1_score(label_list, predicted_list, average='weighted')

    print(f'=> Test F1: {f1_test}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--window_width', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--save_folder', type=str, required=True)
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    train()
    evaluate(model, test_loader, -1, mode='best')
