'''
    classifier delete version
    embedding -> KNN or SVC clustering
'''

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
from models.unsupervised.unsupervised_CAGE import CAGE

import matplotlib.pyplot as plt

def analyze_embeddings(embeddings, labels, save_dir):
    from scipy.spatial.distance import cdist
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    distances = cdist(embeddings, embeddings, metric='cosine')
    
    with open(os.path.join(save_dir, 'embedding_distance_analysis.txt'), 'w') as f:
        f.write ("Embedding Distance Analysis\n")
        f.write("=" * 50 + "\n\n")
        
        intra_distances = []
        inter_distances = []
        
        for label in np.unique(labels):
            mask = labels == label
            intra_dist = distances[mask][:, mask].mean()
            inter_dist = distances[mask][:, ~mask].mean()
            ratio = intra_dist / inter_dist
            
            f.write(f"Label {label}:\n")
            f.write(f"- Intra-class mean distance: {intra_dist:.4f}\n")
            f.write(f"- Inter-class mean distance: {inter_dist:.4f}\n")
            f.write(f"- Ratio (intra/inter): {ratio:.4f}\n\n")
            
            intra_distances.append(intra_dist)
            inter_distances.append(inter_dist)
        
        intra_distances = np.array(intra_distances)
        inter_distances = np.array(inter_distances)
        
        f.write("\nOverall Statistics:\n")
        f.write(f"Mean intra-class distance: {intra_distances.mean():.4f} (±{intra_distances.std():.4f})\n")
        f.write(f"Mean inter-class distance: {inter_distances.mean():.4f} (±{inter_distances.std():.4f})\n")
        f.write(f"Mean ratio (intra/inter): {(intra_distances/inter_distances).mean():.4f}\n")

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
    
    model = CAGE(n_feat // 2, n_cls, proj_dim, args.num_encoders, args.use_skip)
    
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
        NO CLASSIFIER. SO THERE IS NOT CLS_LOSS HERE
    '''
def train_step(model, optimizer, x_accel, x_gyro):
    with tf.GradientTape() as tape:
        ssl_output, (f_accel, f_gyro) = model(x_accel, x_gyro, return_feat=True, training=True)
        
        ssl_output = tf.clip_by_value(ssl_output, 1e-7, 1.0 - 1e-7)
        
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
    
    return np.concatenate(embeddings_list, axis=0), np.array(labels_list)
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
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_path, 'roc_curve.png'))
    plt.close()

def plot_confusion_matrix_heatmap(conf_matrix, save_path):
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
            
            loss, ssl_output = train_step(model, optimizer, x_accel, x_gyro)
            
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
        
        write_scalar_summary(writer, 'Train/Loss', float(epoch_loss), epoch)
        write_scalar_summary(writer, 'Train/Accuracy_ssl', ssl_acc, epoch)
        
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
    
    
    # ------------------- visualization ROC, loss fig, confusion matrix, pca -------------
    
    plot_training_progress(epoch_losses, epoch_ssl_accuracies, args.save_folder)
    plot_roc_curve(test_embeddings, test_labels, knn, args.save_folder)
    plot_confusion_matrix_heatmap(conf_matrix, args.save_folder)
    plot_embeddings_pca(test_embeddings, test_labels, args.save_folder)

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

    val_acc = np.mean(val_predictions == val_labels) * 100
    val_f1 = f1_score(val_labels, val_predictions, average='weighted')
    val_matrix = confusion_matrix(val_labels, val_predictions)
    
    test_acc = np.mean(test_predictions == test_labels) * 100
    test_f1 = f1_score(test_labels, test_predictions, average='weighted')
    test_matrix = confusion_matrix(test_labels, test_predictions)
    
    result = open(os.path.join(args.save_folder, 'result'), 'w+')
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
    result.write(classification_report(test_labels, test_predictions,
                                   zero_division=1))
    
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