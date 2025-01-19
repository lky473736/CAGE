'''
    classifier delete version + NT_Xent_loss + triplet loss 
    embedding -> KNN clustering
'''

import os
from sklearn.manifold import TSNE
import argparse
import time
import random
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm, trange
from sklearn.neighbors import KNeighborsClassifier

from dataset.HAR_dataset import HARDataset
from utils.logger import initialize_logger, record_result, create_tensorboard_writer, write_scalar_summary
from configs import args, dict_to_markdown
from sklearn.metrics import classification_report
from models.unsupervised.NTXent_triplet_CAGE import CAGE, nt_xent_loss, triplet_loss

import matplotlib.pyplot as plt

def analyze_embeddings(embeddings, labels, save_dir) :
    from scipy.spatial.distance import cdist
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    distances = cdist(embeddings, embeddings, metric='cosine')
    
    with open(os.path.join(save_dir, 'embedding_distance_analysis.txt'), 'w') as f :
        f.write ("Embedding Distance Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        intra_distances = []
        inter_distances = []
        
        for label in np.unique(labels) :
            mask = labels == label
            intra_dist = distances[mask][ :, mask].mean()
            inter_dist = distances[mask][ :, ~mask].mean()
            ratio = intra_dist / inter_dist
            
            f.write(f"Label {label} :\n")
            f.write(f"- Intra-class mean distance : {intra_dist :.4f}\n")
            f.write(f"- Inter-class mean distance : {inter_dist :.4f}\n")
            f.write(f"- Ratio (intra/inter) : {ratio :.4f}\n\n")
            
            intra_distances.append(intra_dist)
            inter_distances.append(inter_dist)
        
        intra_distances = np.array(intra_distances)
        inter_distances = np.array(inter_distances)
        
        f.write("\nOverall Statistics :\n")
        f.write(f"Mean intra-class distance : {intra_distances.mean() :.4f} (±{intra_distances.std() :.4f})\n")
        f.write(f"Mean inter-class distance : {inter_distances.mean() :.4f} (±{inter_distances.std() :.4f})\n")
        f.write(f"Mean ratio (intra/inter) : {(intra_distances/inter_distances).mean() :.4f}\n")

def visualize_split_embeddings(accel_embeddings, 
                               gyro_embeddings, 
                               labels, save_dir)  :
    tsne = TSNE(n_components=2, random_state=42)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    accel_tsne = tsne.fit_transform(accel_embeddings)
    scatter1 = ax1.scatter(accel_tsne[ :, 0], accel_tsne[ :, 1],
                          c=labels, cmap='tab20', alpha=0.6)
    ax1.set_title('Accelerometer Embeddings')
    plt.colorbar(scatter1, ax=ax1)
    
    gyro_tsne = tsne.fit_transform(gyro_embeddings)
    scatter2 = ax2.scatter(gyro_tsne[ :, 0], gyro_tsne[ :, 1],
                          c=labels, cmap='tab20', alpha=0.6)
    ax2.set_title('Gyroscope Embeddings')
    plt.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 
                             'split_embeddings.png'), dpi=300, bbox_inches='tight')
    plt.close()

def calculate_metrics(predictions, labels)  :
    from sklearn.metrics import classification_report, confusion_matrix
    
    report = classification_report(labels, predictions, 
                                   output_dict=True, zero_division=True)
    conf_matrix = confusion_matrix(labels, predictions)
    
    return report, conf_matrix

def set_seed(seed)  :
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
def get_model(n_feat, n_cls, weights_path=None) :
    if args.seed :
        set_seed(args.seed)
    
    model = CAGE(n_feat // 2, proj_dim=args.proj_dim, loss_type=args.loss_type)
    
    if weights_path :
        model.load_weights(weights_path)
    elif args.load_model != '' :
        pre_trained_model = tf.keras.models.load_model(args.load_model)
        for i, layer in enumerate(model.layers) :
            if 'classifier' not in layer.name :
                layer.set_weights(pre_trained_model.layers[i].get_weights())
    
    try :
        optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        )
    except :
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    
    initial_learning_rate = args.learning_rate
    decay_steps = args.epochs
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate,
        decay_steps,
        t_mul=2.0,  
        m_mul=0.9, 
        alpha=0.01  
    )
    
    return model, optimizer, lr_schedule

# def get_model(n_feat, n_cls, weights_path=None)  :
#     if args.seed :
#         set_seed(args.seed)
    
#     if args.lambda_ssl > 0 :
#         proj_dim = args.proj_dim
#     else :
#         proj_dim = 0
    
#     model = CAGE(n_feat // 2, n_cls, proj_dim)
    
#     if weights_path :
#         model.load_weights(weights_path)
#     elif args.load_model != '' :
#         pre_trained_model = tf.keras.models.load_model(args.load_model)
#         for i, layer in enumerate(model.layers) :
#             if 'classifier' not in layer.name :
#                 layer.set_weights(pre_trained_model.layers[i].get_weights())
    
#     try :
#         optimizer = tf.keras.optimizers.experimental.AdamW(
#             learning_rate=args.learning_rate,
#             weight_decay=args.weight_decay
#         )
#     except :
#         optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    
#     lr_schedule = lambda epoch : args.learning_rate * (0.5 ** (epoch // 20))
    
#     return model, optimizer, lr_schedule

'''
    all changes -> because of triplet loss
    solve that
'''

# -------- hard negative triplet loss ---------
def get_hard_negatives(anchor_features, all_features, k=10) : 
    distances = tf.reduce_sum(tf.square(
        anchor_features[:, tf.newaxis] - all_features), axis=2)
    _, hard_negative_indices = tf.nn.top_k(-distances, k=k)
    return hard_negative_indices

@tf.function
def train_step(model, optimizer, x_accel, x_gyro):
    '''
        NO CLASSIFIER. SO THERE IS NOT CLS_LOSS HERE
    '''
    with tf.GradientTape() as tape:
        if model.loss_type == 'nt_xent': # nt_xent
            f_accel, f_gyro = model.encode(x_accel, x_gyro, training=True)
            total_loss = nt_xent_loss(f_accel, f_gyro, temperature=args.temperature)
            ssl_output = tf.matmul(tf.math.l2_normalize(f_accel, axis=1), 
                                 tf.math.l2_normalize(f_gyro, axis=1), 
                                 transpose_b=True) / model.temperature
            
            '''
                temperature
            '''
            
        elif model.loss_type == 'triplet' :  # triplet (hard version)
            batch_size = tf.shape(x_accel)[0]
            
            f_anchor, _ = model.call(x_accel, x_accel, training=True)
            f_positive, _ = model.call(x_accel, x_gyro, training=True)
            
            all_features, _ = model.call(x_accel, x_gyro, training=True)
            hard_neg_idx = get_hard_negatives(f_anchor, all_features)
            f_negative = tf.gather(all_features, hard_neg_idx[:, 0])  # most hardest!
            
            total_loss = triplet_loss(f_anchor, f_positive, f_negative, margin=args.margin)
            ssl_output = tf.eye(batch_size)
            
        else : # default
            
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return total_loss, ssl_output

# @tf.function
# def train_step(model, optimizer, x_accel, x_gyro) :
#     '''
#         NO CLASSIFIER. SO THERE IS NOT CLS_LOSS HERE
#     '''
#     with tf.GradientTape() as tape  :
#         ssl_output, (f_accel, f_gyro) = model(x_accel, x_gyro, return_feat=True, training=True)
        
#         ssl_labels = tf.range(tf.shape(ssl_output)[0])
#         ssl_loss_1 = tf.reduce_mean(
#             tf.keras.losses.sparse_categorical_crossentropy(
#                 ssl_labels, ssl_output, from_logits=True
#             )
#         )
        
#         ssl_loss_2 = tf.reduce_mean(
#             tf.keras.losses.sparse_categorical_crossentropy(
#                 ssl_labels, tf.transpose(ssl_output), from_logits=True
#             )
#         )
        
#         total_loss = (ssl_loss_1 + ssl_loss_2) / 2
    
#     gradients = tape.gradient(total_loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     return total_loss, ssl_output

def get_embeddings(model, dataset, n_device)  :
    embeddings_list = []
    labels_list = []
    #
    #
    
    for data, labels in dataset  :
        x_accel = data[ :,  :3 * n_device,  :]
        x_gyro = data[ :, 3 * n_device :,  :]
        _, (f_accel, f_gyro) = model(x_accel, x_gyro, return_feat=True, training=False)
        
        embeddings_list.append(tf.concat([f_accel, f_gyro], axis=1).numpy())
        labels_list.extend(labels.numpy())
    
    return np.concatenate(embeddings_list, axis=0), np.array(labels_list)

def visualize_embeddings(embeddings, labels, save_dir, 
                         prefix='')  :
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[ :, 0], embeddings_2d[ :, 1],
                         c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f'{prefix} Embeddings t-SNE Visualization')
    plt.savefig(os.path.join(save_dir, f'{prefix.lower()}_tsne.jpg'), 
                dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------------
def train()  :
    print (" -------- training... -------- ")
    train_dataset = train_set.make_tf_dataset(
        batch_size=args.batch_size,
        shuffle=True
    ).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = val_set.make_tf_dataset(
        batch_size=args.batch_size,
        shuffle=False
    ).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = test_set.make_tf_dataset(
        batch_size=args.batch_size,
        shuffle=False
    ).prefetch(tf.data.AUTOTUNE)
    
    model, optimizer, lr_schedule = get_model(n_feat, n_cls)
    n_device = n_feat // 6
    
    best_loss = float('inf')
    best_epoch = 0
    
    writer = create_tensorboard_writer(args.save_folder + '/run')
    
    # -------------------------
    # only learning embedding, definitly NO CLASSIFIER HERE
    
    for epoch in trange(args.epochs, desc='Training_epoch')  :
        total_loss = 0
        ssl_labels_list = []
        ssl_preds_list = []
        
        for data, labels in tqdm(train_dataset, desc='Training_batch')  :
            x_accel = data[ :,  :3 * n_device,  :]
            x_gyro = data[ :, 3 * n_device :,  :]
            
            loss, ssl_output = train_step(model, optimizer, x_accel, x_gyro)
            
            batch_size = tf.shape(data)[0]
            total_loss += loss * tf.cast(batch_size, tf.float32)
            
            ssl_labels = tf.range(batch_size)
            ssl_preds = tf.argmax(ssl_output, axis=1)
            ssl_labels_list.append(ssl_labels)
            ssl_preds_list.append(ssl_preds)
        
        if not args.pretrain  :
            optimizer.learning_rate = lr_schedule(epoch)
        
        total_num = len(train_set)
        epoch_loss = total_loss / tf.cast(total_num, tf.float32)
        
        ssl_labels = tf.concat(ssl_labels_list, 0).numpy()
        ssl_preds = tf.concat(ssl_preds_list, 0).numpy()
        ssl_acc = np.mean(ssl_preds == ssl_labels) * 100
        
        logger.info(
            f'Epoch : [{epoch}/{args.epochs}] - '
            f'loss :{float(epoch_loss) :.4f}, '
            f'ssl acc : {ssl_acc :.2f}%'
        )
        
        write_scalar_summary(writer, 'Train/Loss', float(epoch_loss), epoch)
        write_scalar_summary(writer, 'Train/Accuracy_ssl', ssl_acc, epoch)
        
        if epoch_loss < best_loss  : # best model found
            best_loss = epoch_loss
            best_epoch = epoch
            model.save_weights(os.path.join(args.save_folder, 'best.weights.h5'))
            logger.info(f"Best model saved at epoch {epoch}")
    
    model.save_weights(os.path.join(args.save_folder, 'final.weights.h5'))
        
    train_embeddings, train_labels = get_embeddings(model, train_dataset, n_device)
    train_accel = train_embeddings[ :,  :64]   # 64 dim
    train_gyro = train_embeddings[ :, 64 :]    # 64 dim
    val_embeddings, val_labels = get_embeddings(model, val_dataset, n_device)
    test_embeddings, test_labels = get_embeddings(model, test_dataset, n_device)

    #  -------------------------- KNN (or SVM kernel cosine) --------------------------------
    knn = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='cosine')
    
    knn.fit(train_embeddings, train_labels)
    
    val_predictions = knn.predict(val_embeddings)
    test_predictions = knn.predict(test_embeddings)
    
    # from scipy.spatial.distance import cdist
    # from sklearn.svm import SVC
    
    # train_sim = 1 - cdist(train_embeddings, train_embeddings, metric='cosine')
    # val_sim = 1 - cdist(val_embeddings, train_embeddings, metric='cosine')
    # test_sim = 1 - cdist(test_embeddings, train_embeddings, metric='cosine')
    
    # svm = SVC(kernel='precomputed')
    # svm.fit(train_sim, train_labels)
    # val_predictions = svm.predict(val_sim)
    # test_predictions = svm.predict(test_sim)

    #  --------------------- visualization tSNE + save log and rst ----------------------------
    save_dir = os.path.join(args.save_folder, 'embedding_analysis')
    os.makedirs(save_dir, exist_ok=True)

    visualize_split_embeddings(train_accel, train_gyro, train_labels, save_dir)
    analyze_embeddings(train_embeddings, train_labels, save_dir)
    report, conf_matrix = calculate_metrics(test_predictions, test_labels)

    with open(os.path.join(save_dir, 'class_performance.txt'), 'w') as f :
        f.write("Class-wise Performance Metrics\n")
        f.write("=" * 50 + "\n\n")
        for label in sorted(report.keys()) :
            if label.isdigit() :
                metrics = report[label]
                f.write(f"Class {label} :\n")
                f.write(f"- Precision : {metrics['precision'] :.4f}\n")
                f.write(f"- Recall : {metrics['recall'] :.4f}\n")
                f.write(f"- F1-score : {metrics['f1-score'] :.4f}\n")
                f.write(f"- Support : {metrics['support']}\n\n")

    val_acc = np.mean(val_predictions == val_labels) * 100
    val_f1 = f1_score(val_labels, val_predictions, average='weighted')
    val_matrix = confusion_matrix(val_labels, val_predictions)
    
    test_acc = np.mean(test_predictions == test_labels) * 100
    test_f1 = f1_score(test_labels, test_predictions, average='weighted')
    test_matrix = confusion_matrix(test_labels, test_predictions)
    
    result = open(os.path.join(args.save_folder, 'result'), 'w+')
    result.write(f"Best model found at epoch {best_epoch}\n\n")
    
    result.write(f"Validation Metrics :\n")
    result.write(f"Accuracy : {val_acc :.2f}%\n")
    result.write(f"F1 Score : {val_f1 :.4f}\n")
    result.write("Confusion Matrix :\n")
    result.write(str(val_matrix) + "\n\n")
    
    result.write(f"Test Metrics :\n")
    result.write(f"Accuracy : {test_acc :.2f}%\n")
    result.write(f"F1 Score : {test_f1 :.4f}\n")
    result.write("Confusion Matrix :\n")
    result.write(str(test_matrix) + "\n\n")
    
    result.write("Classification Report :\n")
    result.write(classification_report(test_labels, test_predictions, zero_division=True))
    
    result.close()
    writer.close()
    
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