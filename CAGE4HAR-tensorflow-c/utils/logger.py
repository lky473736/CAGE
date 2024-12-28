from __future__ import division
import tensorflow as tf
import logging
import numpy as np
import os

def initialize_logger(file_dir):
    """
    Initialize a logger that prints results to a file.
    
    Args:
        file_dir (str): Directory path for the log file
    
    Returns:
        logger: Configured logging object
    """
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def record_result(result, epoch, best_acc, best_f1, c_mat):
    """ 
    Record evaluation results.
    
    Args:
        result: File object to write results to
        epoch (int): Current epoch number or special value (-1: test best, -2: test final, -3: final validation)
        best_acc (float): Best accuracy achieved
        best_f1 (float): Best F1 score achieved
        c_mat (np.ndarray): Confusion matrix
    """
    if epoch >= 0:
        result.write('Best validation epoch | accuracy: {:.2f}%, F1: {:.4f} (at epoch {})\n'.format(
            best_acc, best_f1, epoch))
    elif epoch == -1:
        result.write('\n\nTest (Best) | accuracy: {:.2f}%, F1: {:.4f}\n'.format(
            best_acc, best_f1))
    elif epoch == -2:
        result.write('\n\nTest (Final) | accuracy: {:.2f}%, F1: {:.4f}\n'.format(
            best_acc, best_f1))
    elif epoch == -3:
        result.write('\n\nFinal validation epoch | accuracy: {:.2f}%, F1: {:.4f}\n'.format(
            best_acc, best_f1))
    
    result.write(np.array2string(c_mat))
    result.flush()

def calculate_metrics(y_true, y_pred, num_classes):
    """
    Calculate accuracy and F1 score using TensorFlow metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
        
    Returns:
        tuple: (accuracy, f1_score)
    """
    accuracy = tf.keras.metrics.Accuracy()
    accuracy.update_state(y_true, y_pred)
    
    # Calculate F1 score
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    
    precision.update_state(y_true, y_pred)
    recall.update_state(y_true, y_pred)
    
    p = precision.result()
    r = recall.result()
    f1 = 2 * (p * r) / (p + r + tf.keras.backend.epsilon())
    
    return accuracy.result().numpy() * 100, f1.numpy()

def get_confusion_matrix(y_true, y_pred, num_classes):
    """
    Calculate confusion matrix using TensorFlow.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
        
    Returns:
        np.ndarray: Confusion matrix
    """
    confusion = tf.zeros((num_classes, num_classes), dtype=tf.int32)
    
    for i in range(len(y_true)):
        confusion = tf.tensor_scatter_nd_add(
            confusion,
            [[y_true[i], y_pred[i]]],
            [1]
        )
    
    return confusion.numpy()