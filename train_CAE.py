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
from models.ConvAE import CAE, MLP_Classifier

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_model(n_feat, n_cls):
    if args.seed:
        set_seed(args.seed)

    model = CAE(n_feat)
    classifier = MLP_Classifier(n_cls)

    try :
        # new (adamw)
        optimizer_model = tf.keras.optimizers.experimental.AdamW(
            learning_rate=args.learning_rate)
        optimizer_classifier = tf.keras.optimizers.experimental.AdamW(
            learning_rate=args.learning_rate)
    except :# old (adam)
        try :
            optimizer_model = tf.keras.optimizers.AdamW(
                learning_rate=args.learning_rate)
            optimizer_classifier = tf.keras.optimizers.AdamW(
                learning_rate=args.learning_rate)
        except:
            print ("-- AdamW is not working, so using standard Adam optimizer --")
            optimizer_model = tf.keras.optimizers.Adam(
                learning_rate=args.learning_rate)
            optimizer_classifier = tf.keras.optimizers.Adam(
                learning_rate=args.learning_rate)
            
     # lr setup
        '''
            at paper : 
            "The learning rate was decayed with a multiplicative factor of 0.5 for every 20 epochs. The model was trained for 200 epochs, and we tested the model’s performance using the parameters of the last epoch."
        '''
    
    try :
        lr_schedule_model = tf.keras.optimizers.schedules.StepDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=25,
            decay_rate=0.8
        )
        lr_schedule_classifier = tf.keras.optimizers.schedules.StepDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=25,
            decay_rate=0.8
        )
        
    except :
        print("legacy learning rate schedule")
        lr_schedule_model = lambda epoch: args.learning_rate * (0.8 ** (epoch // 25))
        lr_schedule_classifier = lambda epoch: args.learning_rate * (0.8 ** (epoch // 25))
    
    if args.load_model != '' :
        model.load_weights(args.load_model)
        
    return [model, classifier], [optimizer_model, optimizer_classifier], [lr_schedule_model, lr_schedule_classifier]

@tf.function
def train_autoencoder_step(model, optimizer, x) :
    """AutoEncoder training section 1 by 1"""
    with tf.GradientTape() as tape :
        x_unsqueezed = tf.expand_dims(x, axis=1)
        output, _ = model(x_unsqueezed, training=True)
        loss = tf.reduce_mean(tf.square(output-x_unsqueezed))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@tf.function
def train_classifier_step(models, optimizers, x, y):
    """classifier training section 1 by 1"""
    model, classifier = models
    optimizer = optimizers[1]
    # optimizer = tf.opti.. <-- 여기 인수로 받아오기
    
    with tf.GradientTape() as tape:
        x_unsqueezed = tf.expand_dims(x, axis=1)
        _, hidden = model(x_unsqueezed, training=False)
        output = classifier(hidden, training=True)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                y, output, from_logits=True
            )
        )
    
    gradients = tape.gradient(loss, classifier.trainable_variables)
    optimizer.apply_gradients(zip(gradients, classifier.trainable_variables))
    
    return loss, output

def evaluate_step(models, x, y) :
    model, classifier = models
    x_unsqueezed = tf.expand_dims(x, axis=1)
    _, hidden = model(x_unsqueezed, training=False)
    output = classifier(hidden, training=False)
    loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            y, output, from_logits=True
        )
    )
    return loss, output

def train():
    train_dataset = train_set.make_tf_dataset(
        batch_size=args.batch_size,
        shuffle=True
    )
    val_dataset = val_set.make_tf_dataset(
        batch_size=args.batch_size,
        shuffle=False
    )
    
    models, optimizers, lr_schedules = get_model(n_feat, n_cls)
    model, classifier = models
    
    best_f1 = 0.0
    best_epoch = 0
    
    result = open(os.path.join(args.save_folder, 'result'), 'w+')
    writer = create_tensorboard_writer(args.save_folder + '/run')
    
    '''
        (1) training AE
        (2) training classifier
    '''
    print("<1> ==> training autoencoder...")
    for epoch in trange(150, desc='Training_autoencoder_epoch'):
        total_loss = 0
        total_num = 0
        
        for x_batch, _ in train_dataset :
            loss = train_autoencoder_step(model, optimizers[0], x_batch)
            total_loss += loss * tf.cast(tf.shape(x_batch)[0], tf.float32)
            total_num += tf.shape(x_batch)[0]
        
        optimizers[0].learning_rate = lr_schedules[0](epoch)
        
        total_loss = total_loss / tf.cast(total_num, tf.float32)
        logger.info(f'Epoch: [{epoch}/150] - loss:{float(total_loss):.4f}')
        write_scalar_summary(writer, 'Train/recon_loss', float(total_loss), epoch)
    
    print("<2> ==> training classifier...")
    model.trainable = False  # freeze autoencoder (because of training below part)
    for epoch in trange(200, desc='Training_classifier_epoch') :
        total_loss = 0
        all_labels = []
        all_preds = []
        
        for x_batch, y_batch in train_dataset : 
            loss, output = train_classifier_step(
                models, optimizers, x_batch, y_batch
            )
            total_loss += loss * tf.cast(tf.shape(x_batch)[0], tf.float32)
            
            preds = tf.argmax(output, axis=1)
            all_labels.extend(y_batch.numpy())
            all_preds.extend(preds.numpy())
        optimizers[1].learning_rate = lr_schedules[1](epoch)        # update lr
        # 
        # 
        # 
        total_num = len(train_set)
        total_loss = total_loss / tf.cast(total_num, tf.float32)
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        
        acc_train = np.mean(all_preds == all_labels) * 100
        f1_train = f1_score(all_labels, all_preds, average='weighted')
        
        logger.info(
            f'Epoch: [{epoch}/200] - '
            f'loss:{float(total_loss):.4f}, '
            f'train acc: {acc_train:.2f}%, '
            f'train F1: {f1_train:.4f}'
        )
        #         write_scalar_summary(writer, 'Train/all_loss_average', loss_all, epoch)
        write_scalar_summary(writer, 'Train/Accuracy_cls', acc_train, epoch)
        write_scalar_summary(writer, 'Train/F1_cls', f1_train, epoch)
        write_scalar_summary(writer, 'Train/cls_loss', float(total_loss), epoch)
        
        # validation
        acc_test, f1_test = evaluate(
            models, val_dataset, epoch,
            is_test=False, writer=writer
        )
        
        if f1_test > best_f1 : # best model save
            best_f1 = f1_test
            best_acc = acc_test
            best_epoch = epoch
            model.save_weights(os.path.join(args.save_folder, 'model_best'))
            classifier.save_weights(os.path.join(args.save_folder, 'classifier_best'))
            c_mat = confusion_matrix(all_labels, all_preds)
    
    model.save_weights(os.path.join(args.save_folder, 'model_final'))
    classifier.save_weights(os.path.join(args.save_folder, 'classifier_final'))
    print('training completed')
    print (f'Best performance at epoch {best_epoch}: '
          f'accuracy = {best_acc:.2f}%, F1 = {best_f1:.4f}')
    print('<confusion matrix>')
    print (c_mat)
    
    record_result(result, best_epoch, best_acc, best_f1, c_mat)
    writer.close()

def evaluate(models, dataset, epoch, is_test=True, mode='best', writer=None) :
    if is_test :
        models[0].load_weights(os.path.join(args.save_folder, f'model_{mode}'))
        models[1].load_weights(os.path.join(args.save_folder, f'classifier_{mode}'))
    
    total_loss = 0
    all_labels = []
    all_preds = []
    
    for x_batch, y_batch in dataset:
        loss, output = evaluate_step(models, x_batch, y_batch)
        total_loss += loss * tf.cast(tf.shape(x_batch)[0], tf.float32)
        
        preds = tf.argmax(output, axis=1)
        all_labels.extend(y_batch.numpy())
        all_preds.extend(preds.numpy())
    
    total_num = len(all_labels)
    total_loss = total_loss / tf.cast(total_num, tf.float32)
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    acc_test = np.mean(all_preds==all_labels) * 100 
    f1_test = f1_score(all_labels, all_preds,average='weighted')

    if is_test == True:
        print (f'=> test acc: {acc_test:.2f}%, test F1: {f1_test:.4f}')
        logger.info (f'=> test acc: {acc_test:.2f}%, test F1: {f1_test:.4f}')
        c_mat = confusion_matrix(all_labels, all_preds)
        result = open(os.path.join(args.save_folder, 'result'), 'a+')
        record_result(result, epoch, acc_test, f1_test, c_mat)
    else :
        logger.info(f'=> val acc: {acc_test:.2f}%, val F1: {f1_test:.4f}')
        logger.info(f'=> loss: {float(total_loss):.4f}')
        
        if writer is not None:
            write_scalar_summary(writer, 'Validation/Accuracy_cls', acc_test, epoch)
            write_scalar_summary(writer, 'Validation/F1_cls', f1_test, epoch)
            write_scalar_summary(writer, 'Validation/Loss_cls', float(total_loss), epoch)

    return acc_test, f1_test

if __name__ == "__main__":
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

    args.save_folder = os.path.join(args.model_path, args.dataset, 'ConvAE', args.trial)
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
    
    models, _, _ = get_model(n_feat, n_cls)
    evaluate(models, test_dataset, -1, mode='best')
    evaluate(models, test_dataset, -2, mode='final')
    evaluate(models, val_dataset, -3, mode='final')