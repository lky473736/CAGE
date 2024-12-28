import os
import argparse
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics
from sklearn.metrics import confusion_matrix, f1_score
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from dataset.HAR_dataset import HARDataset
from utils.logger import initialize_logger, record_result
from configs import args, dict_to_markdown

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_model(input_shape, n_cls):
    set_seed(args.seed)

    if args.model == 'BaselineCNN':
        from models.Baseline_CNN import build_baseline_cnn
        model = build_baseline_cnn(input_shape, n_cls)
    elif args.model == 'DeepConvLSTM':
        from models.DeepConvLSTM import build_deep_conv_lstm
        model = build_deep_conv_lstm(input_shape, n_cls)
    elif args.model == 'LSTMConvNet':
        from models.LSTM_CNN import build_lstm_cnn
        model = build_lstm_cnn(input_shape, n_cls)
    elif args.model == 'EarlyFusion':
        from models.CAGE import build_cage_early_fusion
        model = build_cage_early_fusion(input_shape, n_cls)
    else:
        raise NotImplementedError(f"{args.model} is not implemented")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=args.learning_rate),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    if args.load_model:
        model.load_weights(args.load_model)

    return model

def train():
    train_data = train_set.get_tf_dataset(args.batch_size, shuffle=True)
    val_data = val_set.get_tf_dataset(args.batch_size, shuffle=False)

    input_shape = (train_set.window_width, train_set.feat_dim)
    model = get_model(input_shape, n_cls)

    checkpoint_path = os.path.join(args.save_folder, 'best.h5')
    callbacks = [
        ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max'),
        LearningRateScheduler(lambda epoch: args.learning_rate * 0.5 ** (epoch // 20))
    ]

    print("==> Training...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Save the final model
    final_path = os.path.join(args.save_folder, 'final.h5')
    model.save(final_path)

    print("Done")
    best_epoch = np.argmax(history.history['val_accuracy'])
    best_acc = np.max(history.history['val_accuracy']) * 100
    best_f1, c_mat = evaluate(model, val_data, is_test=False)

    print(f"Best performance achieved at epoch {best_epoch}, best acc: {best_acc:.2f}%, best F1: {best_f1:.4f}")
    print(c_mat)
    record_result(open(os.path.join(args.save_folder, 'result'), 'w+'), best_epoch, best_acc, best_f1, c_mat)

def evaluate(model, data, is_test=True, mode='best'):
    if is_test:
        model.load_weights(os.path.join(args.save_folder, f'{mode}.h5'))

    true_labels = []
    predictions = []

    for x, y in data:
        preds = model.predict(x)
        true_labels.extend(y.numpy())
        predictions.extend(np.argmax(preds, axis=1))

    true_labels = np.array(true_labels)
    predictions = np.array(predictions)

    acc = (predictions == true_labels).mean() * 100
    f1 = f1_score(true_labels, predictions, average='weighted')
    c_mat = confusion_matrix(true_labels, predictions)

    if is_test:
        print(f"Test acc: {acc:.2f}%, Test F1: {f1:.4f}")
    else:
        print(f"Validation acc: {acc:.2f}%, Validation F1: {f1:.4f}")

    return f1, c_mat

if __name__ == "__main__":
    print(dict_to_markdown(vars(args)))

    train_set = HARDataset(dataset=args.dataset, split='train', window_width=args.window_width, clean=args.no_clean, include_null=args.no_null, use_portion=args.train_portion)
    val_set = HARDataset(dataset=args.dataset, split='val', window_width=args.window_width, clean=args.no_clean, include_null=args.no_null)
    test_set = HARDataset(dataset=args.dataset, split='test', window_width=args.window_width, clean=args.no_clean, include_null=args.no_null)

    if args.normalize:
        train_set.normalize(train_set.mean, train_set.std)
        val_set.normalize(train_set.mean, train_set.std)
        test_set.normalize(train_set.mean, train_set.std)

    n_cls = train_set.n_actions

    args.save_folder = os.path.join(args.model_path, args.dataset, args.model, args.trial)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    logger = initialize_logger(os.path.join(args.save_folder, 'train.log'))

    train()

    test_data = test_set.get_tf_dataset(args.batch_size, shuffle=False)
    model = get_model((train_set.window_width, train_set.feat_dim), n_cls)
    evaluate(model, test_data, is_test=True, mode='best')
    evaluate(model, test_data, is_test=True, mode='final')
