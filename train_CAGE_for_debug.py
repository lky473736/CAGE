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
best_epoch = 0

def set_seed(seed): 
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_model(n_feat, n_cls, weights_path=None):
    if args.seed:
        set_seed(args.seed)
    
    if args.lambda_ssl > 0:
        proj_dim = args.proj_dim
    else:
        proj_dim = 0
    
    model = CAGE(n_feat // 2, n_cls, proj_dim)
    
    # 가중치 로드 부분 수정
    if weights_path:
        print(f"[DEBUG] Attempting to load weights from: {weights_path}")
        if os.path.exists(weights_path):
            try:
                model.load_weights(weights_path)
                print(f"[DEBUG] Successfully loaded weights from {weights_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load weights: {e}")
                raise
        else:
            print(f"[ERROR] Weights file does not exist: {weights_path}")
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
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
            print("AdamW not available, using standard Adam optimizer")
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=args.learning_rate
            )
    
    try:
        lr_schedule = tf.keras.optimizers.schedules.StepDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=20,
            decay_rate=0.5
        )
    except:
        print("----- using legacy learning rate schedule-- ")
        lr_schedule = lambda epoch: args.learning_rate * (0.5 ** (epoch // 20))
    
    return model, optimizer, lr_schedule

@tf.function
def train_step(model, optimizer, x_accel, x_gyro, labels):
    with tf.GradientTape() as tape:
        ssl_output, cls_output, (_, _) = model(x_accel, x_gyro, return_feat=True, training=True)
        
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
        
        # Classification loss
        cls_loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels, cls_output, from_logits=True
            )
        )
        
        # Total loss
        total_loss = args.lambda_ssl * ssl_loss + args.lambda_cls * cls_loss
    
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
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
    
    cls_loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            labels, cls_output, from_logits=True
        )
    )
    
    return ssl_loss, cls_loss, ssl_output, cls_output, f_accel, f_gyro

def train():
    global best_epoch
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
    n_device = n_feat // 6  # (accel, gyro) * (x, y, z)
   
    best_f1 = 0.0
    best_epoch = 0
   
    result = open(os.path.join(args.save_folder, 'result'), 'w+')
    writer = create_tensorboard_writer(args.save_folder + '/run')
   
    print("==> training...")
    for epoch in trange(args.epochs, desc='Training_epoch'):
        # First epoch weight saving logic
        if epoch == 1:
            model.save_weights(os.path.join(args.save_folder, 'first.weights.h5'))
            first_val_acc, first_val_f1, first_val_mat = evaluate(
                model, val_dataset, epoch, n_device,
                is_test=False, writer=writer, return_matrix=True
            )
            result.write(f"First model validation acc: {first_val_acc:.2f}%, F1: {first_val_f1:.4f} (epoch 1)\n")
            result.write("First model validation confusion matrix:\n")
            result.write(str(first_val_mat) + "\n\n")
           
        # Training epoch logic
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
       
        if not args.pretrain:
            optimizer.learning_rate = lr_schedule(epoch)
       
        total_num = len(train_set)
        total_loss = total_loss / tf.cast(total_num, tf.float32)
       
        ssl_labels = tf.concat(ssl_labels_list, 0).numpy()
        ssl_preds = tf.concat(ssl_preds_list, 0).numpy()
        ssl_acc = np.mean(ssl_preds == ssl_labels) * 100
       
        cls_labels = tf.concat(cls_labels_list, 0).numpy()
        cls_preds = tf.concat(cls_preds_list, 0).numpy()
        cls_acc = np.mean(cls_preds == cls_labels) * 100
        cls_f1 = f1_score(cls_labels, cls_preds, average='weighted')
        train_matrix = confusion_matrix(cls_labels, cls_preds)
       
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
       
        # Best model saving logic with detailed logging
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_acc = val_acc
            best_epoch = epoch
            best_val_matrix = val_matrix
            best_model_path = os.path.join(args.save_folder, 'best.weights.h5')
            model.save_weights(best_model_path)
            best_train_matrix = train_matrix
            logger.info(f"[BEST MODEL] Updated at epoch {epoch}. F1: {best_f1:.4f}")

    # Final model saving
    final_model_path = os.path.join(args.save_folder, 'final.weights.h5')
    model.save_weights(final_model_path)
    
    # Validation and test evaluation
    final_val_acc, final_val_f1, final_val_matrix = evaluate(
        model, val_dataset, args.epochs, n_device,
        is_test=False, writer=writer, return_matrix=True
    )

    # Evaluating best model on test set
    best_test_model, _, _ = get_model(n_feat, n_cls)
    best_test_model.load_weights(best_model_path)
    best_test_acc, best_test_f1, best_test_matrix = evaluate(
        best_test_model, test_dataset, best_epoch, n_device,
        is_test=True, mode='best', writer=writer, return_matrix=True
    )

    # Evaluating final model on test set
    final_test_model, _, _ = get_model(n_feat, n_cls)
    final_test_model.load_weights(final_model_path)
    final_test_acc, final_test_f1, final_test_matrix = evaluate(
        final_test_model, test_dataset, args.epochs, n_device,
        is_test=True, mode='final', writer=writer, return_matrix=True
    )

    # Detailed results writing
    result.write(f"Best model validation acc: {best_acc:.2f}%, F1: {best_f1:.4f} (epoch {best_epoch})\n")
    result.write(f"Best model test acc: {best_test_acc:.2f}%, F1: {best_test_f1:.4f}\n")
    result.write(f"Final model validation acc: {final_val_acc:.2f}%, F1: {final_val_f1:.4f}\n")
    result.write(f"Final model test acc: {final_test_acc:.2f}%, F1: {final_test_f1:.4f}\n")

    # Console output
    print('\n[RESULTS SUMMARY]')
    print(f'Best Model (epoch {best_epoch}):')
    print(f'  Validation: acc {best_acc:.2f}%, F1 {best_f1:.4f}')
    print(f'  Test: acc {best_test_acc:.2f}%, F1 {best_test_f1:.4f}')
    print(f'Final Model:')
    print(f'  Validation: acc {final_val_acc:.2f}%, F1 {final_val_f1:.4f}')
    print(f'  Test: acc {final_test_acc:.2f}%, F1 {final_test_f1:.4f}')

    result.close()
    writer.close()
def evaluate(model, dataset, epoch, n_device, is_test=True, mode='best', writer=None, return_matrix=False):
    """
    모델 평가 함수 - 상세 디버깅 및 로깅 포함
    """
    print(f"\n[EVALUATE] Mode: {mode}, Is Test: {is_test}")
    
    # 웨이트 파일 경로 설정
    weight_file = os.path.join(args.save_folder, f'{mode}.weights.h5')
    
    # 웨이트 파일 존재 및 유효성 확인
    if is_test:
        if not os.path.exists(weight_file):
            print(f"[ERROR] 웨이트 파일 없음: {weight_file}")
            return 0, 0, None
        
        file_size = os.path.getsize(weight_file)
        print(f"[DEBUG] 웨이트 파일: {weight_file}")
        print(f"[DEBUG] 파일 크기: {file_size} bytes")
        
        try:
            # 웨이트 로딩
            model.load_weights(weight_file)
            print(f"[SUCCESS] {mode} 모델 웨이트 로딩 완료")
        except Exception as e:
            print(f"[CRITICAL] 웨이트 로딩 실패: {e}")
            return 0, 0, None
    
    # 평가 변수 초기화
    ssl_total_loss = 0
    cls_total_loss = 0
    ssl_labels_list = []
    ssl_preds_list = []
    cls_labels_list = []
    cls_preds_list = []
    total_num = 0
    
    # 데이터셋 순회 및 평가
    for data, labels in dataset:
        x_accel = data[:, :3 * n_device, :]
        x_gyro = data[:, 3 * n_device:, :]
        
        ssl_loss, cls_loss, ssl_output, cls_output, _, _ = evaluate_step(
            model, x_accel, x_gyro, labels
        )
        
        batch_size = tf.shape(data)[0]
        ssl_total_loss += ssl_loss * tf.cast(batch_size, tf.float32)
        cls_total_loss += cls_loss * tf.cast(batch_size, tf.float32)
        total_num += batch_size
        
        # 예측 및 레이블 수집
        ssl_labels = tf.range(batch_size)
        ssl_preds = tf.argmax(ssl_output, axis=1)
        ssl_labels_list.append(ssl_labels)
        ssl_preds_list.append(ssl_preds)
        
        cls_preds = tf.argmax(cls_output, axis=1)
        cls_labels_list.append(labels)
        cls_preds_list.append(cls_preds)
    
    # 손실 계산
    ssl_total_loss = ssl_total_loss / tf.cast(total_num, tf.float32)
    cls_total_loss = cls_total_loss / tf.cast(total_num, tf.float32)
    
    # 레이블 및 예측 통합
    ssl_labels = tf.concat(ssl_labels_list, 0).numpy()
    ssl_preds = tf.concat(ssl_preds_list, 0).numpy()
    ssl_acc = np.mean(ssl_preds == ssl_labels) * 100
    
    cls_labels = tf.concat(cls_labels_list, 0).numpy()
    cls_preds = tf.concat(cls_preds_list, 0).numpy()
    
    # 성능 지표 계산
    cls_acc = np.mean(cls_preds == cls_labels) * 100
    cls_f1 = f1_score(cls_labels, cls_preds, average='weighted')
    
    # 혼동 행렬 계산
    c_mat = confusion_matrix(cls_labels, cls_preds)
    
    # 결과 출력
    print(f"[RESULT] {mode.upper()} 모델")
    print(f"정확도: {cls_acc:.2f}%, F1 점수: {cls_f1:.4f}")
    print(f"SSL 정확도: {ssl_acc:.2f}%")
    
    # 결과 파일 기록
    result_filename = f'results_{mode}_model.txt'
    with open(os.path.join(args.save_folder, result_filename), 'w') as f:
        f.write(f"Classification Accuracy: {cls_acc:.2f}%\n")
        f.write(f"Classification F1 Score: {cls_f1:.4f}\n")
        f.write(f"SSL Accuracy: {ssl_acc:.2f}%\n")
        f.write("Confusion Matrix:\n")
        f.write(str(c_mat))
    
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
    
    print("[DEBUG] Evaluating Best Model")
    best_model, _, _ = get_model(n_feat, n_cls)
    best_weights_path = os.path.join(args.save_folder, 'best.weights.h5')
    best_model.load_weights(best_weights_path)
    evaluate(best_model, test_dataset, best_epoch, n_device, mode='best')

    print("[DEBUG] Evaluating Final Model")
    final_model, _, _ = get_model(n_feat, n_cls)
    final_weights_path = os.path.join(args.save_folder, 'final.weights.h5')
    final_model.load_weights(final_weights_path)
    evaluate(final_model, test_dataset, args.epochs, n_device, mode='final') 

    print("[DEBUG] Evaluating First Model")
    first_model, _, _ = get_model(n_feat, n_cls)
    first_weights_path = os.path.join(args.save_folder, 'first.weights.h5')
    first_model.load_weights(first_weights_path)
    evaluate(first_model, test_dataset, 1, n_device, mode='first')

