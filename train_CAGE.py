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
from models.CAGE import CAGE

def set_seed(seed) : 
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_model(n_feat, n_cls):
    if args.seed:
        set_seed(args.seed)
    
    if args.lambda_ssl > 0 :
        proj_dim = args.proj_dim
    else:
        proj_dim = 0
    
    model = CAGE(n_feat // 2, n_cls, proj_dim)
    
    if args.load_model != '':
        pre_trained_model = tf.keras.models.load_model(args.load_model)
        for i, layer in enumerate(model.layers):
            if 'classifier' not in layer.name : # copy weight (definitly need)
                layer.set_weights(pre_trained_model.layers[i].get_weights())
    
    try:
        optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        )
    except:
        try:
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay
            )
        except:
            print ("AdamW not available, using standard Adam optimizer") # <- also CAE, baseline!!
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=args.learning_rate
            )
    try :
        lr_schedule = tf.keras.optimizers.schedules.StepDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=20,
            decay_rate=0.5
        )
        
    except :
        print ("----- using legacy learning rate schedule-- ")
        lr_schedule = lambda epoch: args.learning_rate * (0.5 ** (epoch // 20))
    
    return model, optimizer, lr_schedule

@tf.function
def train_step(model, optimizer, x_accel, x_gyro, labels):
    with tf.GradientTape() as tape:
        ssl_output, cls_output = model(x_accel, x_gyro, training=True)
        
        # supervised-learning loss (contrastive loss) 
        '''
        2개의 encoder 사이의 loss (둘이 가까워지게)
        paper 
        "In this article, we propose an SSL task that connects the features extracted from the accelerometer and the gyroscope. Our method is inspired by the contrastive language-image pretraining (CLIP) [40], which suggests that given the text and image pairs, discriminative features can be retrieved with the contrastive training of the multimodal embedding. "
        '''
        ssl_labels = tf.range(tf.shape(ssl_output)[0])
        ssl_loss_1 = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                ssl_labels, ssl_output, from_logits=True
            )
        )
        ssl_loss_2 = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                ssl_labels, tf.transpose(ssl_output), from_logits=True
            )
        )
        ssl_loss = (ssl_loss_1 + ssl_loss_2) / 2
        
        # cls_loss == classification loss (classifier)
        cls_loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels, cls_output, from_logits=True
            )
        )
        
        '''
            Ltotal = Lcls + λLssl
        '''
        total_loss = args.lambda_ssl * ssl_loss + args.lambda_cls * cls_loss
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # update
    
    return total_loss, ssl_output, cls_output

def evaluate_step(model, x_accel, x_gyro, labels):
    """Evaluation step"""
    ssl_output, cls_output = model(x_accel, x_gyro, training=False)
    
    # SSL loss
    ssl_labels = tf.range(tf.shape(ssl_output)[0])
    ssl_loss_1 = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            ssl_labels, ssl_output, from_logits=True
        )
    )
    ssl_loss_2 = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            ssl_labels, tf.transpose(ssl_output), from_logits=True
        )
    )
    ssl_loss = (ssl_loss_1 + ssl_loss_2) / 2
    
    cls_loss = tf.reduce_mean( # classifier loss
        tf.keras.losses.sparse_categorical_crossentropy(
            labels, cls_output, from_logits=True
        )
    )
    
    return ssl_loss, cls_loss, ssl_output, cls_output

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
    n_device = n_feat // 6  # (accel, gyro) * (x, y, z)
    
    best_f1 = 0.0
    best_epoch = 0
    
    result = open(os.path.join(args.save_folder, 'result'), 'w+')
    writer = create_tensorboard_writer(args.save_folder + '/run')
    
    print("==> training...")
    for epoch in trange(args.epochs, desc='Training_epoch'):
        total_loss = 0
        ssl_labels_list = []
        ssl_preds_list = []
        cls_labels_list = []
        cls_preds_list = []
        
        for data, labels in tqdm(train_dataset, desc='Training_batch'):
            x_accel = data[:, :3 * n_device, :]
            x_gyro = data[:, 3 * n_device:, :]
            
            loss, ssl_output, cls_output = train_step(
                model, optimizer, x_accel, x_gyro, labels
            )
            
            batch_size = tf.shape(data)[0]
            total_loss += loss * tf.cast(batch_size, tf.float32)
            
            ssl_labels = tf.range(batch_size)
            ssl_preds = tf.argmax(ssl_output, axis=1)
            ssl_labels_list.append(ssl_labels)
            ssl_preds_list.append(ssl_preds)
            cls_preds = tf.argmax(cls_output, axis=1)
            cls_labels_list.append(labels)
            cls_preds_list.append(cls_preds)
        
        if not args.pretrain : # update lr
            optimizer.learning_rate = lr_schedule(epoch)
        
        total_num = len(train_set)
        total_loss = total_loss / tf.cast(total_num, tf.float32)
        
        # --------------------------------------------
        
        ssl_labels = tf.concat(ssl_labels_list, 0).numpy()
        ssl_preds = tf.concat(ssl_preds_list, 0).numpy()
        ssl_acc = np.mean(ssl_preds == ssl_labels) * 100
        
        cls_labels = tf.concat(cls_labels_list, 0).numpy()
        cls_preds = tf.concat(cls_preds_list, 0).numpy()
        cls_acc = np.mean(cls_preds == cls_labels) * 100
        cls_f1 = f1_score(cls_labels, cls_preds, average='weighted')
        
        # ---------------------------------------------
        
        logger.info(
            f'Epoch: [{epoch}/{args.epochs}] - '
            f'loss:{float(total_loss):.4f}, '
            f'train acc: {cls_acc:.2f}%, '
            f'train F1: {cls_f1:.4f}, '
            f'ssl acc: {ssl_acc:.2f}%'
        )
        
        write_scalar_summary(writer, 'Train/Accuracy_cls', cls_acc, epoch)
        write_scalar_summary(writer, 'Train/F1_cls', cls_f1, epoch)
        write_scalar_summary(writer, 'Train/Accuracy_ssl', ssl_acc, epoch)
        write_scalar_summary(writer, 'Train/Loss', float(total_loss), epoch)
#             
# 
# 
# 

        val_acc, val_f1 = evaluate(
            model, val_dataset, epoch, n_device,
            is_test=False, writer=writer
        )
        
        
        if args.pretrain and epoch % 50 == 0 :
            model.save_weights(os.path.join(args.save_folder, f'epoch{epoch}.weights.h5'))
            # save pretrained
        
        if val_f1 > best_f1 :  # save best model
            best_f1 = val_f1
            best_acc = val_acc
            best_epoch = epoch
            model.save_weights(os.path.join(args.save_folder, 'best.weights.h5'))
            c_mat = confusion_matrix(cls_labels, cls_preds)
    
    model.save_weights(os.path.join(args.save_folder, 'final.weights.h5')) # final model
    
    print ('-------------- training completed --------------')
    print(f'best performance at epoch {best_epoch}: '
          f'accuracy = {best_acc:.2f}%, F1 = {best_f1:.4f}')
    print('confusion Matrix:')
    print(c_mat)
    
    record_result(result, best_epoch, best_acc, best_f1, c_mat)
    writer.close()

def evaluate(model, dataset, epoch, n_device, is_test=True, mode='best', writer=None):
    if is_test:
        model.load_weights(os.path.join(args.save_folder, f'{mode}.weights.h5'))
    
    ssl_total_loss = 0
    cls_total_loss = 0
    ssl_labels_list = []
    ssl_preds_list = []
    cls_labels_list = []
    cls_preds_list = []
    
    for data, labels in dataset :
        x_accel = data[:, :3 * n_device, :]
        x_gyro = data[:, 3 * n_device:, :]
        
        ssl_loss, cls_loss, ssl_output, cls_output = evaluate_step(
            model, x_accel, x_gyro, labels
        )
        
        batch_size = tf.shape(data)[0]
        ssl_total_loss += ssl_loss * tf.cast(batch_size, tf.float32)
        cls_total_loss += cls_loss * tf.cast(batch_size, tf.float32)
        
        ssl_labels = tf.range(batch_size)
        ssl_preds = tf.argmax(ssl_output, axis=1)
        ssl_labels_list.append(ssl_labels)
        ssl_preds_list.append(ssl_preds)
        
        cls_preds = tf.argmax(cls_output, axis=1)
        cls_labels_list.append(labels)
        cls_preds_list.append(cls_preds)
    
    total_num = sum(tf.shape(labels)[0].numpy() for _, labels in dataset)
    ssl_total_loss = ssl_total_loss / tf.cast(total_num, tf.float32)
    cls_total_loss = cls_total_loss / tf.cast(total_num, tf.float32)
    
    ssl_labels = tf.concat(ssl_labels_list, 0).numpy()
    ssl_preds = tf.concat(ssl_preds_list, 0).numpy()
    ssl_acc = np.mean(ssl_preds == ssl_labels) * 100
    
    cls_labels = tf.concat(cls_labels_list, 0).numpy()
    cls_preds = tf.concat(cls_preds_list, 0).numpy()
    cls_acc = np.mean(cls_preds == cls_labels) * 100
    cls_f1 = f1_score(cls_labels, cls_preds, average='weighted')
    
    if is_test == True :
        print(f'=> test acc: {cls_acc:.2f}%, test F1: {cls_f1:.4f} / '
              f'ssl acc: {ssl_acc:.2f}%')
        
        logger.info(f'=> test acc: {cls_acc:.2f}%, test F1: {cls_f1:.4f} / '
                   f'ssl acc: {ssl_acc:.2f}%')
        
        c_mat = confusion_matrix(cls_labels, cls_preds)
        result = open(os.path.join(args.save_folder, 'result'), 'a+')
        record_result(result, epoch, cls_acc, cls_f1, c_mat)
        
    else :
        logger.info(f'=> val acc (cls): {cls_acc:.2f}%, val F1 (cls): {cls_f1:.4f} / '
                   f'val acc (ssl): {ssl_acc:.2f}%')
        
        logger.info(f'=> cls_loss: {float(cls_total_loss):.4f} / '
                   f'ssl_loss: {float(ssl_total_loss):.4f}')
        
        if writer is not None :
            write_scalar_summary(writer, 'Validation/Accuracy_cls', cls_acc, epoch)
            write_scalar_summary(writer, 'Validation/F1_cls', cls_f1, epoch)
            write_scalar_summary(writer, 'Validation/Accuracy_ssl', ssl_acc, epoch)
            write_scalar_summary(writer, 'Validation/Loss_cls', 
                               float(cls_total_loss), epoch)
            write_scalar_summary(writer, 'Validation/Loss_ssl',
                               float(ssl_total_loss), epoch)
    
    return cls_acc, cls_f1

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

    args.save_folder = os.path.join(args.model_path, args.dataset, 'CAGE', args.trial)
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
    n_device = n_feat // 6  # (accel, gyro) * (x, y, z)
    
    evaluate(model, test_dataset, -1, n_device, mode='best') 
    evaluate(model, test_dataset, -2, n_device, mode='final')
    evaluate(model, val_dataset, -3, n_device, mode='final')