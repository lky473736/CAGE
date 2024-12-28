import os
import argparse
import time
import random
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from dataset.HAR_dataset import HARDataset
from utils.logger import initialize_logger, record_result
from configs import args, dict_to_markdown

# Set device
use_cuda = tf.test.is_gpu_available()
device = '/GPU:0' if use_cuda else '/CPU:0'

def set_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)

def get_model(n_feat, n_cls):
    if args.seed:
        set_seed(args.seed)

    if args.lambda_ssl > 0:
        proj_dim = args.proj_dim
    else:
        proj_dim = 0
    
    model = CAGE(n_feat // 2, n_cls, proj_dim)
    if args.load_model != '':
        model = load_model(args.load_model)
    
    optimizer = Adam(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    return model, optimizer

def train():
    train_loader = HARDataset(dataset=args.dataset, split='train', window_width=args.window_width, 
                              clean=not args.no_clean, include_null=not args.no_null, use_portion=args.train_portion)
    val_loader = HARDataset(dataset=args.dataset, split='val', window_width=args.window_width, 
                            clean=not args.no_clean, include_null=not args.no_null)

    model, optimizer = get_model(n_feat, n_cls)

    n_device = n_feat // 6 # (accel, gyro) * (x, y, z)

    criterion_cls = CategoricalCrossentropy()
    best_f1 = 0.0
    best_epoch = 0

    print("==> training...")
    for epoch in range(args.epochs):
        total_loss = 0
        ssl_gt, ssl_pred, cls_gt, cls_pred = [], [], [], []
        for data, label in train_loader:
            data = tf.convert_to_tensor(data, dtype=tf.float32)
            label = tf.convert_to_tensor(label, dtype=tf.int64)

            accel_x = data[:, :3 * n_device, :]
            gyro_x = data[:, 3 * n_device:, :]
            ssl_output, cls_output = model([accel_x, gyro_x])
            ssl_label = tf.range(tf.shape(ssl_output)[0])

            ssl_loss = (criterion_cls(ssl_output, ssl_label) + criterion_cls(tf.transpose(ssl_output), ssl_label)) / 2
            cls_loss = criterion_cls(cls_output, label)
            loss = ssl_loss * args.lambda_ssl + cls_loss * args.lambda_cls

            optimizer.minimize(loss, model.trainable_variables)

            total_loss += loss.numpy()
            ssl_pred.append(tf.argmax(ssl_output, axis=1).numpy())
            cls_pred.append(tf.argmax(cls_output, axis=1).numpy())
            ssl_gt.append(ssl_label.numpy())
            cls_gt.append(label.numpy())

        total_loss /= len(train_loader)

        label = np.concatenate(cls_gt)
        predicted = np.concatenate(cls_pred)
        acc_train = (predicted == label).sum() * 100.0 / len(label)
        f1_train = f1_score(label, predicted, average='weighted')

        label2 = np.concatenate(ssl_gt)
        predicted2 = np.concatenate(ssl_pred)
        acc_train2 = (predicted2 == label2).sum() * 100.0 / len(label2)

        logger.info(f'Epoch: [{epoch+1}/{args.epochs}] - loss:{total_loss:.4f}, train acc: {acc_train:.2f}%, train F1: {f1_train:.4f} | ssl acc: {acc_train2:.2f}%')

        writer.add_scalar('Train/Accuracy_cls', acc_train, epoch)
        writer.add_scalar('Train/F1_cls', f1_train, epoch)
        writer.add_scalar('Train/Accuracy_ssl', acc_train2, epoch)
        writer.add_scalar('Train/Loss', total_loss, epoch)

        acc_test, f1_test = evaluate(model, val_loader, epoch, is_test=False)
        if epoch % 50 == 0:
            model.save(f"{args.save_folder}/epoch{epoch}.h5")

        if f1_test > best_f1:
            best_f1 = f1_test
            best_epoch = epoch
            model.save(os.path.join(args.save_folder, 'best.h5'))

    model.save(os.path.join(args.save_folder, 'final.h5'))
    print(f'Done. Best performance at epoch {best_epoch}, best acc: {best_acc:.2f}%, best F1: {best_f1:.4f}')

def evaluate(model, eval_loader, epoch, is_test=True, mode='best'):
    if is_test:
        model = load_model(os.path.join(args.save_folder, f"{mode}.h5"))

    model.evaluate(eval_loader)

    criterion_cls = CategoricalCrossentropy()
    n_device = n_feat // 6

    ssl_gt, ssl_pred, cls_gt, cls_pred = [], [], [], []
    for data, label in eval_loader:
        data = tf.convert_to_tensor(data, dtype=tf.float32)
        label = tf.convert_to_tensor(label, dtype=tf.int64)

        accel_x = data[:, :3 * n_device, :]
        gyro_x = data[:, 3 * n_device:, :]
        ssl_output, cls_output = model([accel_x, gyro_x])
        ssl_label = tf.range(tf.shape(ssl_output)[0])

        ssl_loss = (criterion_cls(ssl_output, ssl_label) + criterion_cls(tf.transpose(ssl_output), ssl_label)) / 2
        cls_loss = criterion_cls(cls_output, label)

        ssl_pred.append(tf.argmax(ssl_output, axis=1).numpy())
        cls_pred.append(tf.argmax(cls_output, axis=1).numpy())
        ssl_gt.append(ssl_label.numpy())
        cls_gt.append(label.numpy())

    ssl_gt = np.concatenate(ssl_gt)
    ssl_pred = np.concatenate(ssl_pred)
    cls_gt = np.concatenate(cls_gt)
    cls_pred = np.concatenate(cls_pred)

    acc_test = (cls_pred == cls_gt).sum() * 100.0 / len(cls_gt)
    f1_test = f1_score(cls_gt, cls_pred, average='weighted')
    acc_test2 = (ssl_pred == ssl_gt).sum() * 100.0 / len(ssl_gt)

    if is_test:
        print(f'=> test acc: {acc_test:.2f}%, test F1: {f1_test:.4f} / ssl acc: {acc_test2:.2f}%')
        c_mat = confusion_matrix(cls_gt, cls_pred)
        record_result(os.path.join(args.save_folder, 'result'), epoch, acc_test, f1_test, c_mat)

    else:
        logger.info(f'=> val acc (cls): {acc_test:.2f}%, val F1 (cls): {f1_test:.4f} / val acc (ssl): {acc_test2:.2f}%')

    return acc_test, f1_test

if __name__ == "__main__":
    print(dict_to_markdown(vars(args)))

    # Load datasets
    train_set = HARDataset(dataset=args.dataset, split='train', window_width=args.window_width, 
                           clean=not args.no_clean, include_null=not args.no_null, use_portion=args.train_portion)
    val_set = HARDataset(dataset=args.dataset, split='val', window_width=args.window_width, 
                         clean=not args.no_clean, include_null=not args.no_null)
    test_set = HARDataset(dataset=args.dataset, split='test', window_width=args.window_width, 
                          clean=not args.no_clean, include_null=not args.no_null)

    if args.normalize:
        train_set.normalize(train_set.mean, train_set.std)
        val_set.normalize(train_set.mean, train_set.std)
        test_set.normalize(train_set.mean, train_set.std)

    n_feat = train_set.feat_dim
    n_cls = train_set.n_actions

    args.save_folder = os.path.join(args.model_path, args.dataset, 'CAGE', args.trial)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    # Logging   
    log_dir = os.path.join(args.save_folder, 'train.log')
    logger = initialize_logger(log_dir)
    writer = TensorBoard(args.save_folder)

    # Train
    train()

    # Test
    test_loader = HARDataset(dataset=args.dataset, split='test', window_width=args.window_width, 
                             clean=not args.no_clean, include_null=not args.no_null)
    evaluate(model, test_loader, -1, mode='best')
    evaluate(model, test_loader, -2, mode='final')
    evaluate(model, val_loader, -3, mode='final') 
