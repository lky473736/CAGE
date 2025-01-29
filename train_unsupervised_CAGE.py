'''
    FOR UNSUPERVISED AND SEMI-SUPERVISED
    classifier delete version
    
    embedding training 
    -> K-Means, DBSCAN, fastcluster, gmm, birch
'''

from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, Birch, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from tensorflow.keras import layers, Model
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA

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
from models.embeddings.embedding_CAGE import CAGE

import matplotlib.pyplot as plt

plt.rcParams.update({
   'font.size': 15,
   'axes.titlesize': 17,
   'axes.labelsize': 14,
   'xtick.labelsize': 13,
   'ytick.labelsize': 13,
   'legend.fontsize': 13,
})

def visualize_similarities(train_embeddings, train_labels, save_dir):
    from scipy.spatial.distance import cdist
    similarities = 1 - cdist(train_embeddings, train_embeddings, metric='cosine')
    
    unique_labels = np.unique(train_labels)
    class_similarities = np.zeros((len(unique_labels), len(unique_labels)))
    
    for i, label1 in enumerate(unique_labels):
        for j, label2 in enumerate(unique_labels):
            mask1 = (train_labels == label1)
            mask2 = (train_labels == label2)
            class_similarities[i,j] = np.mean(similarities[mask1][:,mask2]) # <-- mean value of cosine simularity
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(class_similarities, 
                xticklabels=unique_labels,
                yticklabels=unique_labels,
                annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title('Class-wise Average Similarities') # <---- heatmap of simularity
    plt.xlabel('Class')
    plt.ylabel('Class')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_similarities.png'))
    plt.close()
    
    plt.figure(figsize=(12, 5))
    
    intra_similarities = [] # <------- same class simularity
    for label in unique_labels:
        mask = (train_labels == label)
        class_sim = similarities[mask][:,mask]
        intra_similarities.extend(class_sim[np.triu_indices(np.sum(mask), k=1)])
    
    inter_similarities = [] # <-------- other class simularity
    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i+1:]:
            mask1 = (train_labels == label1)
            mask2 = (train_labels == label2)
            inter_similarities.extend(similarities[mask1][:,mask2].flatten())
    
    plt.subplot(1, 2, 1)
    plt.hist(intra_similarities, bins=50, alpha=0.5, label='Intra-class')
    plt.hist(inter_similarities, bins=50, alpha=0.5, label='Inter-class')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.title('Similarity Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    distances = cdist(train_embeddings, train_embeddings, metric='euclidean')
    plt.scatter(distances.flatten(), similarities.flatten(), alpha=0.1)
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Cosine Similarity')
    plt.title('Distance vs Similarity')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'similarity_analysis.png'))
    plt.close()

def visualize_similarity_embedding_relation(train_embeddings, train_labels, save_dir):
    from scipy.spatial.distance import cdist
    
    tsne = TSNE(n_components=2, random_state=42) # 2D
    embeddings_2d = tsne.fit_transform(train_embeddings)
    
    similarities = 1 - cdist(train_embeddings, train_embeddings, metric='cosine')
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=train_labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title('t-SNE Embedding')
    
    plt.subplot(1, 3, 2)
    tsne_distances = cdist(embeddings_2d, embeddings_2d, metric='euclidean')
    plt.scatter(tsne_distances.flatten(), similarities.flatten(), alpha=0.1)
    plt.xlabel('t-SNE Distance')
    plt.ylabel('Original Cosine Similarity')
    plt.title('t-SNE Distance vs Similarity')
    
    #### neighbor's relationship of cosine simularity
    plt.subplot(1, 3, 3)
    k = 5   # <-------- what num of neigh?
    for i in range (len(train_embeddings)) :
        nearest = np.argsort(similarities[i])[-k-1:-1]
        for j in nearest:
            plt.plot([embeddings_2d[i,0], embeddings_2d[j,0]],
                    [embeddings_2d[i,1], embeddings_2d[j,1]],
                    'gray', alpha=0.1)
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=train_labels, cmap='tab10', alpha=0.6)
    plt.title(f'Top {k} Similar Connections')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'embedding_similarity_relation.png'))
    plt.close()

def visualize_split_embeddings(accel_embeddings, 
                               gyro_embeddings, 
                               labels, save_dir) :
    tsne = TSNE(n_components=2, random_state=42)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    accel_tsne = tsne.fit_transform(accel_embeddings)
    scatter1 = ax1.scatter(accel_tsne[:, 0], accel_tsne[:, 1],
                          c=labels, cmap='tab20', alpha=0.6)
    ax1.set_title('Accelerometer Embeddings')
    plt.colorbar(scatter1, ax=ax1)
    
    gyro_tsne = tsne.fit_transform(gyro_embeddings)
    scatter2 = ax2.scatter(gyro_tsne[:, 0], gyro_tsne[:, 1],
                          c=labels, cmap='tab20', alpha=0.6)
    ax2.set_title('Gyroscope Embeddings')
    plt.colorbar(scatter2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 
                             'split_embeddings.png'), dpi=300, bbox_inches='tight')
    plt.close()

def calculate_metrics(predictions, labels) :
    from sklearn.metrics import classification_report, confusion_matrix
    
    report = classification_report(labels, predictions, 
                                   output_dict=True,
                                   zero_division=1)
    conf_matrix = confusion_matrix(labels, predictions)
    
    return report, conf_matrix

def set_seed(seed) :
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_model(n_feat, n_cls, weights_path=None) :
    if args.seed:
        set_seed(args.seed)
    
    if args.lambda_ssl > 0:
        proj_dim = args.proj_dim
    else:
        proj_dim = 0
    
    # model = CAGE(n_feat // 2, n_cls, proj_dim, args.num_encoders, args.use_skip)
    model = CAGE(
        n_feat // 2, 
        n_cls, 
        proj_dim=proj_dim,
        encoder_type=args.encoder_type,
        num_heads=args.num_heads if args.encoder_type == 'transformer' else None,
        num_encoders=args.num_encoders if args.encoder_type == 'default' else None,
        use_skip=args.use_skip if args.encoder_type == 'default' else None
    )
    
    if weights_path:
        model.load_weights(weights_path)
    elif args.load_model != '':
        pre_trained_model = tf.keras.models.load_model(args.load_model)
        for i, layer in enumerate(model.layers):
            if 'classifier' not in layer.name:
                layer.set_weights(pre_trained_model.layers[i].get_weights())
    
    try:
        optimizer = tf.keras.optimizers.experimental.AdamW(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay
        )
    except:
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    
    lr_schedule = lambda epoch: args.learning_rate * (0.5 ** (epoch // 20))
    
    return model, optimizer, lr_schedule

'''
    CAUTION!
    NO CLASSIFIER. SO THERE IS NOT CLS_LOSS HERE
'''
@tf.function
def train_step(model, optimizer, x_accel, x_gyro):
    with tf.GradientTape() as tape:
        ssl_output, (f_accel, f_gyro) = model(x_accel, x_gyro, return_feat=True, training=True)
        
        ssl_output = tf.math.log_softmax(ssl_output, axis=-1)
        
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
        
        total_loss = tf.where(
            tf.math.is_finite(ssl_loss_1 + ssl_loss_2),
            (ssl_loss_1 + ssl_loss_2) / 2,
            0.0
        )

    gradients = tape.gradient(total_loss, model.trainable_variables)
    gradients = [
        tf.clip_by_norm(g, 1.0) if g is not None else g 
        for g in gradients
    ]
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss, ssl_output

def get_embeddings(model, dataset, n_device):
    embeddings_list = []
    labels_list = []
    
    for data, labels in dataset:
        x_accel = data[:, :3 * n_device, :]
        x_gyro = data[:, 3 * n_device:, :]
        _, (f_accel, f_gyro) = model(x_accel, x_gyro, return_feat=True, training=False)
        
        embeddings = tf.concat([f_accel, f_gyro], axis=1).numpy()
        '''
            ValueError: Input X contains NaN.
            KNeighborsClassifier does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values
            
            
            -> solve this so that add checking condition if is nan
        '''
        if np.any(np.isnan(embeddings)) :
            print ("WARNING: NaN VALUES DETECTED")
            embeddings = np.nan_to_num(embeddings, 0)
        
        embeddings_list.append(embeddings)
        labels_list.extend(labels.numpy())
    
    embeddings = np.concatenate(embeddings_list, axis=0)
    
    if args.use_pca : # YOU USE PCA?
        if not hasattr(get_embeddings, 'pca') : 
            n_components = min(args.pca_components, 
                               embeddings.shape[1]) # PCA dimensionss
            get_embeddings.pca = PCA(n_components=n_components)
            embeddings = get_embeddings.pca.fit_transform(embeddings)
            print (f"{n_components} components: {np.sum(get_embeddings.pca.explained_variance_ratio_):.4f}")
        else :
            embeddings = get_embeddings.pca.transform(embeddings)
    
    return embeddings, np.array(labels_list)

    #     '''
    #         ValueError: Input X contains NaN.
    #         KNeighborsClassifier does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values
            
            
    #         -> solve this so that add checking condition if is nan
    #     '''
    #     if np.any(np.isnan(embeddings)) :
    #         print ("WARNING: NaN VALUES DETECTED")
    #         embeddings = np.nan_to_num(embeddings, 0)
        
    #     embeddings_list.append(embeddings)
    #     labels_list.extend(labels.numpy())
    
    # return np.concatenate(embeddings_list, axis=0), np.array(labels_list)

def visualize_embeddings(embeddings, labels, save_dir, 
                         prefix='') :
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                         c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f'{prefix} Embeddings t-SNE Visualization')
    plt.savefig(os.path.join(save_dir, f'{prefix.lower()}_tsne.jpg'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_training_progress(epoch_losses, epoch_ssl_accuracies, save_path):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_losses)
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epoch_ssl_accuracies)
    plt.title('SSL Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_progress.png'))
    plt.close()

def plot_roc_curve(test_embeddings, test_labels, knn, save_path):
    from sklearn.metrics import roc_curve, auc

    test_prob = knn.predict_proba(test_embeddings)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(test_labels, test_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', 
             lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', 
             lw=2, 
             linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_path, 'roc_curve.png'))
    plt.close()

def plot_confusion_matrix_heatmap(conf_matrix, save_path) :
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix_heatmap.png'))
    plt.close()

def plot_embeddings_pca(test_embeddings, test_labels, save_path):
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(test_embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                          c=test_labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('PCA of Test Embeddings')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'embeddings_pca.png'))
    plt.close()

# -----------------------------------

def apply_clustering(embeddings, cluster_type='kmeans', **kwargs):
    if cluster_type == 'kmeans':
        clusterer = KMeans(n_clusters=kwargs.get('n_clusters', 2))
    elif cluster_type == 'dbscan':
        clusterer = DBSCAN(
            eps=kwargs.get('eps', 0.5),
            min_samples=kwargs.get('min_samples', 5)
        )
    # elif cluster_type == 'spectral':
    #     clusterer = SpectralClustering(
    #         n_clusters=kwargs.get('n_clusters', 2),
    #         affinity=kwargs.get('affinity', 'rbf')
    #     )
    elif cluster_type == 'birch':
        clusterer = Birch(
            n_clusters=kwargs.get('n_clusters', 2),
            threshold=kwargs.get('threshold', 0.5),
            branching_factor=kwargs.get('branching_factor', 50)  
        )
    elif cluster_type == 'gmm':
        clusterer = GaussianMixture(
            n_components=kwargs.get('n_clusters', 2),
            max_iter=50
        )
    elif cluster_type == 'fastcluster':
        try:
            batch_size = 1000
            n_samples = len(embeddings)
            predictions = np.zeros(n_samples)
            
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                batch_embeddings = embeddings[i:batch_end]
                
                clusterer = AgglomerativeClustering(
                    n_clusters=kwargs.get('n_clusters', 2),
                    linkage=kwargs.get('linkage', 'average')
                )
                predictions[i:batch_end] = clusterer.fit_predict(batch_embeddings)
                
            return predictions
        except Exception as e:
            print(f"FastCluster error: {e}")
            exit()
            # fallback to K-means
    
    if hasattr(clusterer, 'fit_predict'):
        return clusterer.fit_predict(embeddings)
    else:
        clusterer.fit(embeddings)
        return clusterer.predict(embeddings)

# --------------------------------
def train() :
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
    
    epoch_losses = []
    epoch_ssl_accuracies = []
    
    # -------------------------
    # only learning embedding, definitly NO CLASSIFIER HERE
    
    for epoch in trange(args.epochs, desc='Training_epoch') :
        total_loss = 0
        ssl_labels_list = []
        ssl_preds_list = []
        
        for data, labels in tqdm(train_dataset, desc='Training_batch') :
            x_accel = data[:, :3 * n_device, :]
            x_gyro = data[:, 3 * n_device:, :]
            
            loss, ssl_output = train_step(model, optimizer, 
                                          x_accel, x_gyro)
            
            batch_size = tf.shape(data)[0]
            total_loss += loss * tf.cast(batch_size, tf.float32)
            
            ssl_labels = tf.range(batch_size)
            ssl_preds = tf.argmax(ssl_output, axis=1)
            ssl_labels_list.append(ssl_labels)
            ssl_preds_list.append(ssl_preds)
            
        
        if not args.pretrain :
            optimizer.learning_rate = lr_schedule(epoch)
        
        total_num = len(train_set)
        epoch_loss = total_loss / tf.cast(total_num, tf.float32)
        
        ssl_labels = tf.concat(ssl_labels_list, 0).numpy()
        ssl_preds = tf.concat(ssl_preds_list, 0).numpy()
        ssl_acc = np.mean(ssl_preds == ssl_labels) * 100
        
        logger.info(
            f'Epoch: [{epoch}/{args.epochs}] - '
            f'loss:{float(epoch_loss):.4f}, '
            f'ssl acc: {ssl_acc:.2f}%'
        )
        
        write_scalar_summary(writer, 'Train/Loss', 
                             float(epoch_loss), epoch)
        write_scalar_summary(writer, 'Train/Accuracy_ssl', 
                             ssl_acc, epoch)
        
        if epoch_loss < best_loss : # best model found
            best_loss = epoch_loss
            best_epoch = epoch
            model.save_weights(os.path.join(args.save_folder, 'best.weights.h5'))
            logger.info(f"Best model saved at epoch {epoch}")
            
        epoch_losses.append(float(epoch_loss))
        epoch_ssl_accuracies.append(ssl_acc)
    
    model.save_weights(os.path.join(args.save_folder, 'final.weights.h5'))
        
    train_embeddings, train_labels = get_embeddings(model, train_dataset, n_device)
    train_accel = train_embeddings[:, :64]   # 64 dim
    train_gyro = train_embeddings[:, 64:]    # 64 dim
    val_embeddings, val_labels = get_embeddings(model, val_dataset, n_device)
    test_embeddings, test_labels = get_embeddings(model, test_dataset, n_device)
    
    save_dir = os.path.join(args.save_folder, 'embedding_analysis')
    os.makedirs(save_dir, exist_ok=True)
    
    # visualization embeddings
    visualize_similarities(train_embeddings, train_labels, save_dir)
    visualize_similarity_embedding_relation(train_embeddings, train_labels, save_dir)

    
    # ------------------- K-Means, DBSCAN, spectral clustering ---------------------------
    
    def get_cluster_labels(true_labels, pred_labels) :
        from scipy.stats import mode
        cluster_labels = np.zeros_like(pred_labels)
        for cluster in range(len(np.unique(pred_labels))):
            mask = (pred_labels == cluster)
            if np.sum(mask) > 0: 
                cluster_labels[mask] = mode(true_labels[mask])[0]
        return cluster_labels

    # for training 
    train_predictions = apply_clustering(
        train_embeddings, 
        cluster_type=args.clustering_method,
        n_clusters=n_cls,
        eps=args.dbscan_eps,
        min_samples=args.dbscan_min_samples,
        affinity=args.spectral_affinity
    )

    # for val
    val_predictions = apply_clustering(
        val_embeddings, 
        cluster_type=args.clustering_method,
        n_clusters=n_cls,
        eps=args.dbscan_eps,
        min_samples=args.dbscan_min_samples,
        affinity=args.spectral_affinity
    )

    # for test
    test_predictions = apply_clustering(
        test_embeddings, 
        cluster_type=args.clustering_method,
        n_clusters=n_cls,
        eps=args.dbscan_eps,
        min_samples=args.dbscan_min_samples,
        affinity=args.spectral_affinity
    )

    train_mapped_predictions = get_cluster_labels(train_labels, train_predictions)
    val_mapped_predictions = get_cluster_labels(val_labels, val_predictions)
    test_mapped_predictions = get_cluster_labels(test_labels, test_predictions)

    #  -------------------------- KNN (or SVM kernel cosine) --------------------------------
    # knn = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='cosine')
    
    # knn.fit(train_embeddings, train_labels)
    
    # val_predictions = knn.predict(val_embeddings)
    # test_predictions = knn.predict(test_embeddings)
    
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

    visualize_split_embeddings(train_accel, train_gyro, train_labels, save_dir)
    # analyze_embeddings(train_embeddings, train_labels, save_dir)
    # report, conf_matrix = calculate_metrics(test_predictions, test_labels)
    
    report, conf_matrix = calculate_metrics(train_mapped_predictions, train_labels)

    
    # ------------------- visualization ROC, loss fig, confusion matrix, pca -------------
    
    plot_training_progress(epoch_losses, 
                           epoch_ssl_accuracies, 
                           args.save_folder)
    # plot_roc_curve(test_embeddings, 
    #                test_labels, knn, 
    #                args.save_folder)
    plot_confusion_matrix_heatmap(confusion_matrix(test_labels, test_mapped_predictions), 
                            args.save_folder)
    plot_embeddings_pca(test_embeddings, 
                        test_labels, 
                        args.save_folder)

    with open(os.path.join(save_dir, 'class_performance.txt'), 'w') as f:
        f.write("Class-wise Performance Metrics\n")
        f.write("=" * 50 + "\n\n")
        for label in sorted(report.keys()):
            if label.isdigit():
                metrics = report[label]
                f.write(f"Class {label}:\n")
                f.write(f"- Precision: {metrics['precision']:.4f}\n")
                f.write(f"- Recall: {metrics['recall']:.4f}\n")
                f.write(f"- F1-score: {metrics['f1-score']:.4f}\n")
                f.write(f"- Support: {metrics['support']}\n\n")

    ######################### KNN, SVM
    # val_acc = np.mean(val_predictions == val_labels) * 100
    # val_f1 = f1_score(val_labels, val_predictions, average='weighted')
    # val_matrix = confusion_matrix(val_labels, val_predictions)
    
    # test_acc = np.mean(test_predictions == test_labels) * 100
    # test_f1 = f1_score(test_labels, test_predictions, average='weighted')
    # test_matrix = confusion_matrix(test_labels, test_predictions)
    
    ######################### K-Means
    val_acc = np.mean(val_mapped_predictions == val_labels) * 100
    val_f1 = f1_score(val_labels, val_mapped_predictions, average='weighted')
    val_matrix = confusion_matrix(val_labels, val_mapped_predictions)
    
    test_acc = np.mean(test_mapped_predictions == test_labels) * 100
    test_f1 = f1_score(test_labels, test_mapped_predictions, average='weighted')
    test_matrix = confusion_matrix(test_labels, test_mapped_predictions)
    
    result = open(os.path.join(args.save_folder, 'result'), 
                  'w+')
    result.write(f"Best model found at epoch {best_epoch}\n\n")
    
    result.write(f"Validation Metrics:\n")
    result.write(f"Accuracy: {val_acc:.2f}%\n")
    result.write(f"F1 Score: {val_f1:.4f}\n")
    result.write("Confusion Matrix:\n")
    result.write(str(val_matrix) + "\n\n")
    
    result.write(f"Test Metrics:\n")
    result.write(f"Accuracy: {test_acc:.2f}%\n")
    result.write(f"F1 Score: {test_f1:.4f}\n")
    result.write("Confusion Matrix:\n")
    result.write(str(test_matrix) + "\n\n")
    
    result.write("Classification Report:\n")
    result.write(classification_report(test_labels, 
                                       test_mapped_predictions, zero_division=1))
    result.close()
    writer.close()
    
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