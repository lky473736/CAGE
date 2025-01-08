import os
from sklearn.manifold import TSNE
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
from sklearn.metrics import classification_report
from models.CAGE import CAGE

import matplotlib.pyplot as plt


def set_seed(seed) : 
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_model(n_feat, n_cls, weights_path=None):
    if args.seed:
        set_seed(args.seed)
    
    if args.lambda_ssl > 0 :
        proj_dim = args.proj_dim
    else:
        proj_dim = 0
    
    model = CAGE(n_feat // 2, n_cls, proj_dim)
    
    # 가중치 로드 부분 수정
    if weights_path:
        model.load_weights(weights_path)
    elif args.load_model != '':
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
        ssl_output, cls_output, (_, _) = model(x_accel, x_gyro, return_feat=True, training=True)
        
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
    ssl_output, cls_output, (f_accel, f_gyro) = model(x_accel, x_gyro, return_feat=True, training=False)
    
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
    
    return ssl_loss, cls_loss, ssl_output, cls_output, f_accel, f_gyro

def train():
   train_dataset = train_set.make_tf_dataset(
       batch_size=args.batch_size,
       shuffle=True
   ).prefetch(tf.data.AUTOTUNE)  # 데이터셋 prefetch 추가
   
   val_dataset = val_set.make_tf_dataset(
       batch_size=args.batch_size,
       shuffle=False
   ).prefetch(tf.data.AUTOTUNE)
   
   test_dataset = test_set.make_tf_dataset(
       batch_size=args.batch_size,
       shuffle=False
   ).prefetch(tf.data.AUTOTUNE)
   
   model, optimizer, lr_schedule = get_model(n_feat, n_cls)
   n_device = n_feat // 6  # (accel, gyro) * (x, y, z)
   
   best_f1 = 0.0
   best_epoch = 0
   
   result = open(os.path.join(args.save_folder, 'result'), 'w+')
   writer = create_tensorboard_writer(args.save_folder + '/run')
   
   print("==> training...")
   for epoch in trange(args.epochs, desc='Training_epoch'):
       if epoch == 1:  # Save model after first epoch
           model.save_weights(os.path.join(args.save_folder, 'first.weights.h5'))
           first_val_acc, first_val_f1, first_val_mat = evaluate(
               model, val_dataset, epoch, n_device,
               is_test=False, writer=writer, return_matrix=True
           )
           result.write(f"First model validation acc: {first_val_acc:.2f}%, F1: {first_val_f1:.4f} (epoch 1)\n")
           result.write("First model validation confusion matrix:\n")
           result.write(str(first_val_mat) + "\n\n")
           
       
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
       
       if not args.pretrain: # update lr
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
       train_matrix = confusion_matrix(cls_labels, cls_preds)
       
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

       val_acc, val_f1, val_matrix = evaluate(
           model, val_dataset, epoch, n_device,
           is_test=False, writer=writer, return_matrix=True
       )
       
       if args.pretrain and epoch % 50 == 0:
           model.save_weights(os.path.join(args.save_folder, f'epoch{epoch}.weights.h5'))
       
    #    if val_f1 > best_f1:  # save best model
    #        best_f1 = val_f1
    #        best_acc = val_acc
    #        best_epoch = epoch
    #        best_val_matrix = val_matrix
    #        model.save_weights(os.path.join(args.save_folder, 'best.weights.h5'))
    #        best_train_matrix = train_matrix
    
       if val_f1 > best_f1:  
        best_f1 = val_f1
        best_acc = val_acc
        best_epoch = epoch
        best_val_matrix = val_matrix
        model.save_weights(os.path.join(args.save_folder, 'best.weights.h5'))
        best_train_matrix = train_matrix
        logger.info(f"Best model updated at epoch {epoch}")

   
   model.save_weights(os.path.join(args.save_folder, 'final.weights.h5'))
   final_val_acc, final_val_f1, final_val_matrix = evaluate(
       model, val_dataset, args.epochs, n_device,
       is_test=False, writer=writer, return_matrix=True
   )

   model, _, _ = get_model(n_feat, n_cls)  
   best_weights_path = os.path.join(args.save_folder, 'best.weights.h5')
   model.load_weights(best_weights_path)
   best_test_acc, best_test_f1, best_test_matrix = evaluate(
        model, test_dataset, best_epoch, n_device,
        is_test=True, writer=writer, return_matrix=True
    )

   model, _, _ = get_model(n_feat, n_cls)  
   final_weights_path = os.path.join(args.save_folder, 'final.weights.h5')
   model.load_weights(final_weights_path)
   final_test_acc, final_test_f1, final_test_matrix = evaluate(
        model, test_dataset, args.epochs, n_device,
        is_test=True, writer=writer, return_matrix=True
    )
   
   result.write(f"Best model validation acc: {best_acc:.2f}%, F1: {best_f1:.4f} (epoch {best_epoch})\n")
   result.write("Best model validation confusion matrix:\n")
   result.write(str(best_val_matrix) + "\n\n")
   
   result.write(f"Best model test acc: {best_test_acc:.2f}%, F1: {best_test_f1:.4f}\n")
   result.write("Best model test confusion matrix:\n")
   result.write(str(best_test_matrix) + "\n\n")
   
   result.write(f"Final model validation acc: {final_val_acc:.2f}%, F1: {final_val_f1:.4f} (epoch {args.epochs})\n")
   result.write("Final model validation confusion matrix:\n")
   result.write(str(final_val_matrix) + "\n\n")
   
   result.write(f"Final model test acc: {final_test_acc:.2f}%, F1: {final_test_f1:.4f}\n")
   result.write("Final model test confusion matrix:\n")
   result.write(str(final_test_matrix))
   
   print('-------------- training completed --------------')
   print(f'Best model (epoch {best_epoch}):')
   print(f'Validation - acc: {best_acc:.2f}%, F1: {best_f1:.4f}')
   print(f'Test - acc: {best_test_acc:.2f}%, F1: {best_test_f1:.4f}')
   print('\nFinal model:')
   print(f'Validation - acc: {final_val_acc:.2f}%, F1: {final_val_f1:.4f}')
   print(f'Test - acc: {final_test_acc:.2f}%, F1: {final_test_f1:.4f}')
   
   result.close()
   writer.close()
   
def calculate_embedding_distances(accel_embeddings, gyro_embeddings, labels):
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    
    intra_distances = []  # distances within same label
    inter_distances = []  # distances between different labels
    label_stats = {}
    
    for label in unique_labels:
        mask = labels == label
        curr_accel = accel_embeddings[mask]
        curr_gyro = gyro_embeddings[mask]
        
        intra_dist = np.mean(np.linalg.norm(curr_accel - curr_gyro, axis=1))
        intra_distances.append(intra_dist)
        
        other_mask = labels != label
        other_accel = accel_embeddings[other_mask]
        other_gyro = gyro_embeddings[other_mask]
        
        inter_dist = np.mean(np.linalg.norm(curr_accel[:, None] - other_gyro, axis=2))
        inter_distances.append(np.mean(inter_dist))
        
        label_stats[label] = {
            'intra_dist': intra_dist,
            'inter_dist': np.mean(inter_dist),
            'ratio': intra_dist / np.mean(inter_dist)
        }
    
    return np.array(intra_distances), np.array(inter_distances), label_stats

def visualize_embeddings(accel_embeddings, gyro_embeddings, labels, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    
    accel_tsne = tsne.fit_transform(accel_embeddings)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    scatter1 = ax1.scatter(accel_tsne[:, 0], accel_tsne[:, 1], 
                          c=labels, cmap='tab10', alpha=0.6)
    ax1.set_title('Accelerometer Embeddings')
    fig.colorbar(scatter1, ax=ax1)
    
    gyro_tsne = tsne.fit_transform(gyro_embeddings)
    scatter2 = ax2.scatter(gyro_tsne[:, 0], gyro_tsne[:, 1], 
                          c=labels, cmap='tab10', alpha=0.6)
    ax2.set_title('Gyroscope Embeddings')
    fig.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_embedding_analysis(save_dir, intra_distances, inter_distances, label_stats, 
                          cls_labels, cls_preds, all_accel_embeddings, all_gyro_embeddings):
    """Save embedding analysis results"""
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'embedding_analysis.txt'), 'w') as f:
        f.write("Embedding Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Overall Statistics:\n")
        f.write(f"Mean Intra-class Distance: {np.mean(intra_distances):.4f} (±{np.std(intra_distances):.4f})\n")
        f.write(f"Mean Inter-class Distance: {np.mean(inter_distances):.4f} (±{np.std(inter_distances):.4f})\n")
        f.write(f"Mean Ratio (Intra/Inter): {np.mean(intra_distances/inter_distances):.4f}\n\n")
        
        f.write("Per-label Statistics:\n")
        for label, stats in label_stats.items():
            f.write(f"\nLabel {label}:\n")
            f.write(f"  Intra-class Distance: {stats['intra_dist']:.4f}\n")
            f.write(f"  Inter-class Distance: {stats['inter_dist']:.4f}\n")
            f.write(f"  Ratio: {stats['ratio']:.4f}\n")
   
def evaluate(model, dataset, epoch, n_device, is_test=True, mode='best', writer=None, return_matrix=False):
    if is_test:
        model.load_weights(os.path.join(args.save_folder, f'{mode}.weights.h5'))
    
    ssl_total_loss = 0
    cls_total_loss = 0
    ssl_labels_list = []
    ssl_preds_list = []
    cls_labels_list = []
    cls_preds_list = []
    accel_embeddings = []
    gyro_embeddings = []
    
    total_num = 0
    
    for data, labels in dataset:
        x_accel = data[:, :3 * n_device, :]
        x_gyro = data[:, 3 * n_device:, :]
        
        ssl_loss, cls_loss, ssl_output, cls_output, f_accel, f_gyro = evaluate_step(
            model, x_accel, x_gyro, labels
        )
        
        batch_size = tf.shape(data)[0]
        ssl_total_loss += ssl_loss * tf.cast(batch_size, tf.float32)
        cls_total_loss += cls_loss * tf.cast(batch_size, tf.float32)
        total_num += batch_size
        
        ssl_labels = tf.range(batch_size)
        ssl_preds = tf.argmax(ssl_output, axis=1)
        ssl_labels_list.append(ssl_labels)
        ssl_preds_list.append(ssl_preds)
        
        cls_preds = tf.argmax(cls_output, axis=1)
        cls_labels_list.append(labels)
        cls_preds_list.append(cls_preds)
        
        accel_embeddings.append(f_accel.numpy())
        gyro_embeddings.append(f_gyro.numpy())
    
    ssl_total_loss = ssl_total_loss / tf.cast(total_num, tf.float32)
    cls_total_loss = cls_total_loss / tf.cast(total_num, tf.float32)
    
    ssl_labels = tf.concat(ssl_labels_list, 0).numpy()
    ssl_preds = tf.concat(ssl_preds_list, 0).numpy()
    ssl_acc = np.mean(ssl_preds == ssl_labels) * 100
    
    cls_labels = tf.concat(cls_labels_list, 0).numpy()
    cls_preds = tf.concat(cls_preds_list, 0).numpy()
    cls_acc = np.mean(cls_preds == cls_labels) * 100
    cls_f1 = f1_score(cls_labels, cls_preds, average='weighted')
    
    all_accel_embeddings = np.concatenate(accel_embeddings, axis=0)
    all_gyro_embeddings = np.concatenate(gyro_embeddings, axis=0)
    c_mat = confusion_matrix(cls_labels, cls_preds)
    
    if is_test:
        print(f'=> test acc: {cls_acc:.2f}%, test F1: {cls_f1:.4f} / '
              f'ssl acc: {ssl_acc:.2f}%')
        
        logger.info(f'=> test acc: {cls_acc:.2f}%, test F1: {cls_f1:.4f} / '
                   f'ssl acc: {ssl_acc:.2f}%')
        
        save_dir = os.path.join(args.save_folder, f'embedding_analysis_{mode}')
        os.makedirs(save_dir, exist_ok=True)
        
        # --------------- 1. t-SNE visualization ---------------
        tsne = TSNE(n_components=2, random_state=42)
        plt.figure(figsize=(12, 5))
        
        accel_tsne = tsne.fit_transform(all_accel_embeddings)
        plt.subplot(121)
        scatter = plt.scatter(accel_tsne[:, 0], accel_tsne[:, 1], 
                            c=cls_labels, cmap='tab10', alpha=0.6)
        plt.title('Accelerometer Embeddings')
        plt.colorbar(scatter)
        
        gyro_tsne = tsne.fit_transform(all_gyro_embeddings)
        plt.subplot(122)
        scatter = plt.scatter(gyro_tsne[:, 0], gyro_tsne[:, 1], 
                            c=cls_labels, cmap='tab10', alpha=0.6)
        plt.title('Gyroscope Embeddings')
        plt.colorbar(scatter)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'tsne_visualization.jpg'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # --------------- 2. Distance analysis ---------------

        unique_labels = np.unique(cls_labels)
        intra_distances = []  # within same label
        inter_distances = []  # between different labels
        
        with open(os.path.join(save_dir, 'embedding_distance_analysis.txt'), 'w') as f:
            f.write("Embedding Distance Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            for label in unique_labels:
                mask = cls_labels == label
                curr_accel = all_accel_embeddings[mask]
                curr_gyro = all_gyro_embeddings[mask]
                
                intra_dist = np.mean(np.linalg.norm(curr_accel - curr_gyro, axis=1))
                intra_distances.append(intra_dist)
                
                other_mask = cls_labels != label
                other_accel = all_accel_embeddings[other_mask]
                other_gyro = all_gyro_embeddings[other_mask]
                inter_dist = np.mean(np.linalg.norm(curr_accel[:, None] - other_gyro, axis=2))
                inter_distances.append(np.mean(inter_dist))
                
                f.write(f"Label {label}:\n")
                f.write(f"- Intra-class mean distance: {intra_dist:.4f}\n")
                f.write(f"- Inter-class mean distance: {inter_dist:.4f}\n")
                f.write(f"- Ratio (intra/inter): {intra_dist/inter_dist:.4f}\n\n")
            
            intra_distances = np.array(intra_distances)
            inter_distances = np.array(inter_distances)
            f.write("\nOverall Statistics:\n")
            f.write(f"Mean intra-class distance: {np.mean(intra_distances):.4f} (±{np.std(intra_distances):.4f})\n")
            f.write(f"Mean inter-class distance: {np.mean(inter_distances):.4f} (±{np.std(inter_distances):.4f})\n")
            f.write(f"Mean ratio (intra/inter): {np.mean(intra_distances/inter_distances):.4f}\n")
        
        plt.figure(figsize=(10, 6))
        plt.hist(intra_distances, alpha=0.5, label='Same Label', bins=20)
        plt.hist(inter_distances, alpha=0.5, label='Different Labels', bins=20)
        plt.xlabel('Euclidean Distance')
        plt.ylabel('Frequency')
        plt.title('Distribution of Embedding Distances')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'distance_distribution.jpg'))
        plt.close()
        
        result_filename = f'results_{mode}_model.txt'
        with open(os.path.join(args.save_folder, result_filename), 'w') as f:
            f.write(f"test acc : {cls_acc:.2f}%\n")
            f.write(f"test f1 : {cls_f1:.4f}\n")
            
            class_report = classification_report(cls_labels, cls_preds, output_dict=True)
            
            f.write("\nPer-class Performance:\n")
            for class_name, metrics in class_report.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    f.write(f"class {class_name}:\n")
                    f.write(f"  Precision: {metrics['precision']:.4f}\n")
                    f.write(f"  Recall: {metrics['recall']:.4f}\n")
                    f.write(f"  F1-Score: {metrics['f1-score']:.4f}\n")
            
            f.write("\nOverall Performance:\n")
            f.write(f"Macro Avg Precision: {class_report['macro avg']['precision']:.4f}\n")
            f.write(f"Macro Avg Recall: {class_report['macro avg']['recall']:.4f}\n")
            f.write(f"Macro Avg F1-Score: {class_report['macro avg']['f1-score']:.4f}\n")

            f.write("\nConfusion Matrix:\n")
            f.write(str(c_mat))
            f.write("\n\n")
            
            f.write("Per-sample Results:\n")
            for i in range(len(cls_labels)):
                f.write(f"[Sample {i+1}]\n")
                f.write(f"Original Label: {cls_labels[i]}\n")
                f.write(f"Predicted Label: {cls_preds[i]}\n")
                f.write(f"Accelerometer Embedding: {all_accel_embeddings[i].tolist()[:10]}\n")
                f.write(f"Gyroscope Embedding: {all_gyro_embeddings[i].tolist()[:10]}\n\n")
                
    else:
        logger.info(f'=> val acc (cls): {cls_acc:.2f}%, val F1 (cls): {cls_f1:.4f} / '
                   f'val acc (ssl): {ssl_acc:.2f}%')
        
        logger.info(f'=> cls_loss: {float(cls_total_loss):.4f} / '
                   f'ssl_loss: {float(ssl_total_loss):.4f}')
        
        if writer is not None:
            write_scalar_summary(writer, 'Validation/Accuracy_cls', cls_acc, epoch)
            write_scalar_summary(writer, 'Validation/F1_cls', cls_f1, epoch)
            write_scalar_summary(writer, 'Validation/Accuracy_ssl', ssl_acc, epoch)
            write_scalar_summary(writer, 'Validation/Loss_cls', 
                               float(cls_total_loss), epoch)
            write_scalar_summary(writer, 'Validation/Loss_ssl',
                               float(ssl_total_loss), epoch)
    
    if return_matrix:
        return cls_acc, cls_f1, c_mat
    return cls_acc, cls_f1

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
    evaluate(model, test_dataset, -3, n_device, mode='first')