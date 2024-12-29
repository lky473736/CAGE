import os
import argparse
import time
import random
import numpy as np
import tensorflow as tf

from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm, trange

from dataset.HAR_dataset import HARDataset
from utils.logger import initialize_logger, record_result, create_tensorboard_writer, write_scalar_summary
from configs import args, dict_to_markdown

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_model(n_feat, n_cls):
    if args.seed:
        set_seed(args.seed)

    if args.model == 'BaselineCNN' :
        from models.Baseline_CNN import Baseline_CNN
        model = Baseline_CNN(n_feat, n_cls, 128)
    elif args.model == 'DeepConvLSTM' :
        from models.DeepConvLSTM import DeepConvLSTM
        model = DeepConvLSTM(n_feat, n_cls)
    elif args.model == 'LSTMConvNet' :
        from models.LSTM_CNN import LSTMConvNet
        model = LSTMConvNet(n_feat, n_cls)
    elif args.model == 'EarlyFusion' :
        from models.CAGE import CAGE_EarlyFusion
        model = CAGE_EarlyFusion(n_feat, n_cls)
    else:
        raise NotImplementedError(f"{args.model} is not implemented")
        
    if args.load_model != '':
        model.load_weights(args.load_model)
    
    try :
        '''
            adamw : https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
            adam : tensorflow 2.0 is not supporting adamw
        '''
        optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        )
    except :
        try : 
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay
            )
        except :
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=args.learning_rate
            )
    
    # lr setup
    try :
        '''
            at paper : 
            "The learning rate was decayed with a multiplicative factor of 0.5 for every 20 epochs. The model was trained for 200 epochs, and we tested the model’s performance using the parameters of the last epoch."
        '''
        lr_schedule = tf.keras.optimizers.schedules.StepDecay( # use step decay for implementing the way of lr scheduling at paper
            initial_learning_rate=args.learning_rate,
            decay_steps=20,
            decay_rate=0.5
        )
    except :
        print ("---- legacy mode ----")
        lr_schedule = lambda epoch: args.learning_rate * (0.5 ** (epoch // 20)) # old way
    
    return model, optimizer, lr_schedule

@tf.function # without this error occured (i dont know why)
def train_step(model, optimizer, x, y) :
    with tf.GradientTape() as tape :
        logits = model(x, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y, logits, from_logits=True)
        loss = tf.reduce_mean(loss)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss, logits

# @tf.function
def evaluate_step(model, x, y) : 
    logits = model(x, training=False)
    
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        y, logits, from_logits=True)
    
    loss = tf.reduce_mean(loss)
    
    return loss, logits

def train() :
    train_dataset = train_set.make_tf_dataset(
        batch_size=args.batch_size, 
        shuffle=True
    )
    val_dataset = val_set.make_tf_dataset(
        batch_size=args.batch_size,
        shuffle=False
    )
    
    model, optimizer, lr_schedule = get_model(n_feat, n_cls)

    best_f1, best_epoch = 0.0, 0
    
    result = open(os.path.join(args.save_folder, 'result'), 'w+')
    writer = create_tensorboard_writer(args.save_folder + '/run')
    
    print("==> training...")
    for epoch in trange(args.epochs, desc='Training_epoch') : # updating lr
        if callable(lr_schedule) : # old
            optimizer.learning_rate = lr_schedule(epoch)
        else : # new (stepdecay)
            optimizer.learning_rate = lr_schedule(epoch)
        
        total_loss = 0 
        all_labels = []
        all_preds = []
        
        for x_batch, y_batch in tqdm(train_dataset, desc='Training_batch') :
            loss, logits = train_step(model, optimizer, x_batch, y_batch)
            total_loss += loss * tf.cast(tf.shape(x_batch)[0], tf.float32)
            
            preds = tf.argmax(logits, axis=1) # prediction 
            all_labels.extend(y_batch.numpy())
            all_preds.extend(preds.numpy())
            
        total_num = len(train_set)
        total_loss = total_loss / tf.cast(total_num, tf.float32)
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        
        acc_train = np.mean(all_preds == all_labels) * 100
        f1_train = f1_score(all_labels, all_preds, average='weighted')
        
        logger.info( # logging
            f'Epoch: [{epoch}/{args.epochs}] - ' 
            f'loss:{float(total_loss):.4f}, '
            f'train acc: {acc_train:.2f}%, '
            f'train F1: {f1_train:.4f}'
        )
        
        write_scalar_summary(writer, 'Train/Accuracy', acc_train, epoch) # tensorboard
        write_scalar_summary(writer, 'Train/F1', f1_train, epoch)
        write_scalar_summary(writer, 'Train/Loss', float(total_loss), epoch)
        write_scalar_summary(writer, 'Train/Learning_Rate', 
                           float(optimizer.learning_rate), epoch)
        
        val_acc, val_f1 = evaluate( # validation
            model, val_dataset, epoch,
            is_test=False, writer=writer
        )
        
        if val_f1 > best_f1 : # best model saving section
            best_f1 = val_f1
            best_acc = val_acc
            best_epoch = epoch
            model.save_weights(os.path.join(args.save_folder, 'best.weights.h5'))
            c_mat = confusion_matrix(all_labels, all_preds)
    model.save_weights(os.path.join(args.save_folder, 'final.weights.h5'))
    
    print ('-------------- Training completed ---------------')
    print (f'Best performance at epoch {best_epoch}: '
          f'accuracy = {best_acc:.2f}%, F1 = {best_f1:.4f}')
    print('<Confusion Matrix>')
    print (c_mat)
    
    record_result(result, best_epoch, best_acc, best_f1, c_mat)
    writer.close()

def evaluate(model, dataset, epoch, is_test=True, mode='best', writer=None) :
    if is_test :
        model.load_weights(os.path.join(args.save_folder, f'{mode}.weights.h5'))
        
    total_loss = 0
    all_labels = []
    all_preds = []
    
    for x_batch, y_batch in dataset :
        loss, logits = evaluate_step(model, x_batch, y_batch)
        total_loss += loss * tf.cast(tf.shape(x_batch)[0], tf.float32)
        preds = tf.argmax(logits, axis=1)
        all_labels.extend(y_batch.numpy())
        all_preds.extend(preds.numpy())
    
    total_num = len(all_labels)
    total_loss = total_loss / tf.cast(total_num, tf.float32)
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    acc_test = np.mean(all_preds == all_labels) * 100
    f1_test = f1_score(all_labels, all_preds, average='weighted')

    if is_test :
        print(f'=> test acc: {acc_test:.2f}%, test F1: {f1_test:.4f}')
        logger.info(f'=> test acc: {acc_test:.2f}%, test F1: {f1_test:.4f}')
        c_mat = confusion_matrix(all_labels, all_preds)
        result = open(os.path.join(args.save_folder, 'result'), 'a+')
        record_result(result, epoch, acc_test, f1_test, c_mat)
        
    else :
        logger.info(f'=> val acc: {acc_test:.2f}%, val F1: {f1_test:.4f}')
        logger.info(f'=> loss: {float(total_loss):.4f}')
        
        if writer is not None :
            write_scalar_summary(writer, 'Validation/Accuracy', acc_test, epoch)
            write_scalar_summary(writer, 'Validation/F1', f1_test, epoch)
            write_scalar_summary(writer, 'Validation/Loss', float(total_loss), epoch)

    return acc_test, f1_test

if __name__ == "__main__" :
    print(dict_to_markdown(vars(args)))

    train_set = HARDataset(
        dataset=args.dataset,
        split='train',
        window_width=args.window_width,
        clean=args.no_clean,
        include_null=args.no_null,
        use_portion=args.train_portion
    )
    
    val_set = HARDataset(
        dataset=args.dataset,
        split='val',
        window_width=args.window_width,
        clean=args.no_clean,
        include_null=args.no_null
    )
    
    test_set = HARDataset(
        dataset=args.dataset,
        split='test',
        window_width=args.window_width,
        clean=args.no_clean,
        include_null=args.no_null
    )

    if args.normalize:
        train_set.normalize(train_set.mean, train_set.std)
        val_set.normalize(train_set.mean, train_set.std)
        test_set.normalize(train_set.mean, train_set.std)

    n_feat = train_set.feat_dim
    n_cls = train_set.n_actions

    args.save_folder = os.path.join(args.model_path, args.dataset, args.model, args.trial)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)
        
    log_dir = os.path.join(args.save_folder, 'train.log')
    logger = initialize_logger(log_dir)
    writer = create_tensorboard_writer(args.save_folder + '/run')

    train()

    test_dataset = test_set.make_tf_dataset(
        batch_size=args.batch_size,
        shuffle=False
    )
    val_dataset = val_set.make_tf_dataset(
        batch_size=args.batch_size,
        shuffle=False
    )
    
    model, _, _ = get_model(n_feat, n_cls)
    
    dummy_batch = next(iter(test_dataset))
    _ = model(dummy_batch[0], training=False)
    
    evaluate(model, test_dataset, -1, mode='best')
    evaluate(model, test_dataset, -2, mode='final')
    evaluate(model, val_dataset, -3, mode='final')