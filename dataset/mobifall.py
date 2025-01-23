from dataset import *
from dataset.dataset_generator import HARDataGenerator
import glob
import numpy as np

# 1/16

class MobiFall(HARDataGenerator):
   def __init__(self, window_length=128, clean=False, fall=True) :
       super(MobiFall, self).__init__()
       self.clean = clean
       self.fall = fall
       self.sampling_rate = 87  # RESTORE to original sampling rate
       self.original_rate = 87  # original sampling rate
       self.WINDOW_LENGTH = window_length
       self.STRIDE = self.WINDOW_LENGTH // 2
       
       #--------------------------------------------------
       
       '''
           DATASET 
           split subjects into train/val/test
           
           # first 11 subjects: all activities but...
           # subjects 12-31: falls only
       '''
       adl_fall_subjects = list(range(1, 12))  # 1-11: both ADL and Falls
       fall_only_subjects = list(range(12, 22)) + list(range(29, 32))  # 12-21, 29-31: fall only

       train_adl_fall = adl_fall_subjects[:8]     # 8 
       val_adl_fall = adl_fall_subjects[8:10]     # 2
       test_adl_fall = adl_fall_subjects[10:]     # 1 

       train_fall_only = fall_only_subjects[:9]    # 9 
       val_fall_only = fall_only_subjects[9:11]    # 2 
       test_fall_only = fall_only_subjects[11:]    # 2 

       train_split = train_adl_fall + train_fall_only
       val_split = val_adl_fall + val_fall_only
       test_split = test_adl_fall + test_fall_only

       print("\nDataset split information:")
       print(f"Train subjects (ADL+Fall): {train_adl_fall}")
       print(f"Train subjects (Fall only): {train_fall_only}")
       print(f"Total train subjects: {train_split}\n")
       
       print(f"Val subjects (ADL+Fall): {val_adl_fall}")
       print(f"Val subjects (Fall only): {val_fall_only}")
       print(f"Total val subjects: {val_split}\n")
       
       print(f"Test subjects (ADL+Fall): {test_adl_fall}")
       print(f"Test subjects (Fall only): {test_fall_only}")
       print(f"Total test subjects: {test_split}\n")
       
       self.datapath = "./data/MobiFall_Dataset"

       # for binary classification, 0 or 1 
       '''
           ADL = 0, Fall = 1
       '''
       self.adl_activities = ['STD', 'WAL', 'JOG', 
                              'JUM', 'STU', 'STN', 
                              'SCH', 'CSI', 'CSO']
       self.fall_activities = ['FOL', 'FKL', 'BSC', 'SDL'] #fall
       self.label2id = {act: 0 for act in self.adl_activities}
       self.label2id.update({act: 1 for act in self.fall_activities})

       self.train_data, self.train_label = self._read_data(train_split)
       self.val_data, self.val_label = self._read_data(val_split)
       self.test_data, self.test_label = self._read_data(test_split)

   def _read_sensor_file(self, filepath) :
    '''
        #Acceleration force along the x y z axes (including gravity).
        #timestamp(ns),x,y,z(m/s^2)
        #Datetime: 05/08/2013 11:20:15
        ##########################################
        #Activity: 9 - CSO - Car Step-out - 6s
        #Subject ID: 2
        #First Name: sub2
        #Last Name: sub2
        #Age: 26
        #Height(cm): 169
        #Weight(kg): 64
        #Gender: Male
        ##########################################


        @DATA
        184458554000, 1.2641385, -1.7525556, 9.490616
        184658824000, 1.6950948, -1.733402, 8.992621
        ...
    '''
    with open(filepath, 'r') as f:
        lines = f.readlines()
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() == '@DATA': 
                data_start = i + 1
                break
        
        data_lines = []
        for line in lines[data_start:]:
            if line.strip():  
                try:
                    values = [float(val.strip()) for val in line.strip().split(',')]
                    if all(np.isfinite(values)):
                        data_lines.append(values)
                except ValueError:
                    continue
        
        df = pd.DataFrame(data_lines)
        df = df.fillna(0)
        
        return df

   def _read_data(self, split):
       data = []
       labels = []
       
       for subject_id in split:
           subject_dir = os.path.join(self.datapath, f"sub{subject_id}")
           if not os.path.exists(subject_dir):
               print(f"Subject directory not found: {subject_dir}")
               continue
               
           if subject_id < 12 :
               adl_dir = os.path.join(subject_dir, "ADL")
               if os.path.exists(adl_dir):
                   for activity in self.adl_activities:
                       activity_dir = os.path.join(adl_dir, activity)
                       if os.path.exists(activity_dir):
                           self._process_activity_data(activity_dir, activity, subject_id, data, labels)
           
           falls_dir = os.path.join(subject_dir, "FALLS")
           if os.path.exists(falls_dir):
               for activity in self.fall_activities:
                   activity_dir = os.path.join(falls_dir, activity)
                   if os.path.exists(activity_dir):
                       self._process_activity_data(activity_dir, activity, subject_id, data, labels)

       if len(data) == 0:
           raise ValueError(f"No valid data was loaded for subjects {split}")
           
       try:
           concat_data = np.concatenate(data, axis=0)
           concat_labels = np.concatenate(labels)

           unique_labels, counts = np.unique(concat_labels, return_counts=True)
           dist = list(zip(unique_labels, counts))
           print(f"Data distribution for split {split}: {dist}")
           return concat_data, concat_labels
       except ValueError as e:
           print(f"Error concatenating data for subjects {split}: {str(e)}")
           return np.array([]), np.array([])
   
   def _process_activity_data(self, activity_dir, activity, subject_id, data, labels):
    acc_pattern = f"{activity}_acc_{subject_id}_*.txt"
    acc_files = glob.glob(os.path.join(activity_dir, acc_pattern))
    
    for acc_file in acc_files:
        try:
            trial_num = acc_file.split('_')[-1]
            gyro_file = os.path.join(activity_dir, f"{activity}_gyro_{subject_id}_{trial_num}") # gyro
            
            if not os.path.exists(gyro_file):
                print(f"Gyro file not found for: {acc_file}")
                continue

            try:
                acc_data = self._read_sensor_file(acc_file)
                acc_data.columns = ['timestamp', 'acc_x', 'acc_y', 'acc_z']
                
                gyro_data = self._read_sensor_file(gyro_file)
                gyro_data.columns = ['timestamp', 'gyro_x', 'gyro_y', 'gyro_z']
            except Exception as e:
                print(f"Error reading sensor files: {str(e)}")
                continue
            
            try:
                # NaN
                merged_data = pd.merge_asof(
                    acc_data.sort_values('timestamp'),
                    gyro_data.sort_values('timestamp'),
                    on='timestamp',
                    direction='nearest',
                    tolerance=1e7,
                    allow_exact_matches=True
                )
                
                merged_data = merged_data.fillna(0)
                
                sensor_data = merged_data[['acc_x', 'acc_y', 'acc_z',
                                        'gyro_x', 'gyro_y', 'gyro_z']].values
            except Exception as e:
                print(f"ERROR merging sensor data: {str(e)}")
                continue
            
            # Removed downsampling
            sensor_data = np.nan_to_num(sensor_data, 0)

            try:
                filtered_data = butterworth_filter(sensor_data, self.sampling_rate) # <---------- neccessary?????
            except Exception as e:
                print(f"Error applying butterworth filter: {str(e)}")
                continue
            
            activity_labels = np.full(len(filtered_data), self.label2id[activity])
            
            try:
                windows_data, windows_labels = self.split_windows(filtered_data, activity_labels) # <- split_seqeunces
                
                if windows_data is not None and len(windows_data) > 0:
                    data.append(windows_data)
                    labels.append(windows_labels)
            except Exception as e:
                print(f"Error splitting windows: {str(e)}")
                continue
                
        except Exception as e:
            print(f"Error processing {acc_file}: {str(e)}")
            continue

if __name__ == "__main__":
   mobifall = MobiFall()
   mobifall.dataset_verbose()
   mobifall.save_split()