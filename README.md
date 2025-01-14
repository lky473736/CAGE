# CAGE
implementation of paper, CAGE4HAR using TensorFlow

- **Gyuyeon Lim (lky473736)**
- making period : 2024.12.01. ~ 2025.01.03.
- testing : 2025.01.01. ~ current

> [!NOTE]
> **This source code is a TensorFlow conversion of the implementation code from <Contrastive Accelerometer–Gyroscope Embedding Model for Human Activity Recognition (https://ieeexplore.ieee.org/document/9961198)>** and is not the official source code. The official source code is implemented in PyTorch and can be found here (https://github.com/quotation2520/CAGE4HAR).


-----

### Basic structure

```bash
CAGE4HAR-main/
├── dataset/
│   ├── HAR_dataset.py     
│   └── dataset_specific/  
├── models/
│   ├── Baseline_CNN.py   
│   ├── CAGE.py          
│   └── ...              
├── utils/
│   └── logger.py         
└── train scripts/
    ├── train_baseline.py 
    ├── train_CAGE.py
    └── train_CAE.py
```

-----

### Conversion points between the original PyTorch implementation and the TensorFlow conversion

```python
# <example 1 : handling dataset>

# PyTorch 
class HARDataset(Dataset):
    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

# TensorFlow 
class HARDataset :
    def make_tf_dataset(self, batch_size=32, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.data, self.label)
        )
        if shuffle:
            dataset = dataset.shuffle(len(self.data))
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
```

```python
# <example 2 : training loopstation>

# PyTorch
optimizer.zero_grad()
loss.backward()
optimizer.step()

# TensorFlow
with tf.GradientTape() as tape :
    loss = compute_loss(model, x, y)
gradients = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

The conversion maintained CAGE4HAR's core functionality while adapting to TensorFlow's ecosystem, transforming PyTorch's eager execution to TensorFlow's graph-based computation. 

- The Dataset implementation was changed from PyTorch's ```__getitem__``` to TensorFlow's tf.data.Dataset API with performance optimizations. 

- The model architecture was converted from nn.Module to keras.Model, changing forward() to call() method and preserving weight initialization strategies. 

- The training pipeline replaced PyTorch's backward() with TensorFlow's GradientTape for gradient computation, and adapted the learning rate scheduling. 

- Model saving/loading switched from .pth to .weights.h5 format, with careful attention to tensor operations. 

- The contrastive learning and dual-branch architecture for accelerometer and gyroscope data were preserved while utilizing TensorFlow's batch processing. 

- TensorFlow-specific optimizations were added, including @tf.function decorator and efficient data loading, along with enhanced error handling. 

- The monitoring system was adapted to use TensorBoard, and version compatibility was implemented with fallback options.

-----

### How to activate CAGE4HAR-tensorflow

**1. Download and prepare datasets**
```bash
# Place datasets in data/
data/
├── UCI HAR Dataset
├── MHEALTHDATASET
├── PAMAP2_Dataset
└── MobiAct_Dataset_v2.0

# Process datasets
python create_dataset.py
```

**2. Train models**
```bash
- models
    - 'BaselineCNN'
    - 'DeepConvLSTM'
    - 'LSTMConvNet'
    - 'EarlyFusion'
    - 'CAE'
    - 'CAGE'
- datasets 
    - 'UCI_HAR'
    - 'WISDM'
    - 'Opportunity'
    - 'USC_HAD'
    - 'PAMAP2'
    - 'mHealth'
    - 'MobiAct'
```

```bash
# Train baseline models
python train_baseline.py --model [BaselineCNN/DeepConvLSTM/LSTMConvNet] --dataset [Dataset]

# Train CAGE model
python train_CAGE.py --dataset [Dataset]

# Train ConvAE model
python train_CAE.py --dataset [Dataset]
```

**Example**
```bash
python train_CAGE.py --dataset PAMAP2 --batch_size 64 --epochs 200 --normalize
```

Results will be saved in `save/{dataset}/{model}/{trial}/`.

------

### Paper Summary

- **summary about paper used by reference during implementing**
    - https://astonishing-antlion-13b.notion.site/Contrastive-Accelerometer-Gyroscope-Embedding-Model-for-Human-Activity-Recognition-162e61796c4280b59b13d7978e33ad8c?pvs=4
    - https://astonishing-antlion-13b.notion.site/CLIP-Learning-Transferable-Visual-Models-From-Natural-Language-Supervision-163e61796c4280ff95e8cf08dc5c759e?pvs=4
