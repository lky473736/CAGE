from dataset import *
from dataset.dataset_generator import HARDataGenerator
import glob
import numpy as np
import pandas as pd
import os
from collections import Counter

# 1/17
# editing

class SisFall(HARDataGenerator):
    def __init__(self, window_length=128, 
                clean=False, 
                fall=True):
        super(SisFall, self).__init__()
        self.clean = clean
        self.fall = fall
        self.sampling_rate = 200  # RESTORE to original sampling rate
        self.original_rate = 200
        self.WINDOW_LENGTH = window_length
        self.STRIDE = self.WINDOW_LENGTH // 2
        
        # SA: adult subjects (SA01-SA23)
        # SE: elderly subjects (SE01-SE15, SE06 has falls)
        adult_subjects = [f'SA{str(i).zfill(2)}' for i in range(1, 24)]  # 23
        elderly_subjects = [f'SE{str(i).zfill(2)}' for i in range(1, 16)]  # 15
        '''
            # adult subjects (~70/15/15)
            # elderly subjects (~70/15/15)
        '''
        
        
        train_sa = adult_subjects[:16]      # 16
        val_sa = adult_subjects[16:19]      # 3
        test_sa = adult_subjects[19:]       # 4
        
        train_se = elderly_subjects[:10]    # 10
        val_se = elderly_subjects[10:12]    # 2
        test_se = elderly_subjects[12:]     # 3
    
        train_split = train_sa + train_se # combining because avoiding biased age
        val_split = val_sa + val_se
        test_split = test_sa + test_se
        
        # --------------------------------------------------=
        print ()
        print ("Dataset split information:")
        print ("Adult subjects (SA):", adult_subjects)
        print("Elderly subjects (SE):", elderly_subjects)
        print(f"\nTrain subjects (SA): {train_sa}")
        print (f"Train subjects (SE): {train_se}")
        print (f"Total train subjects: {train_split}")
        
        print(f"\nVal subjects (SA): {val_sa}")
        print (f"Val subjects (SE): {val_se}")
        print (f"Total val subjects: {val_split}")
        
        print (f"\nTest subjects (SA): {test_sa}")
        print (f"Test subjects (SE): {test_se}")
        print(f"Total test subjects: {test_split}")
        
        print (f"Train subjects: {train_split}")
        print (f"Val subjects: {val_split}")
        print (f"Test subjects: {test_split}\n")
        
        # --------------------------------------------------=
        self.datapath = "./data/SisFall_Dataset"

        '''
        Each file contains nine columns and a different number of rows depending on the test length.
        
        1st column is the acceleration data in the X axis measured by the sensor ADXL345.
        2nd column is the acceleration data in the Y axis measured by the sensor ADXL345.
        3rd column is the acceleration data in the Z axis measured by the sensor ADXL345.

        4th column is the rotation data in the X axis measured by the sensor ITG3200.
        5th column is the rotation data in the Y axis measured by the sensor ITG3200.
        6th column is the rotation data in the Z axis measured by the sensor ITG3200.

        7th column is the acceleration data in the X axis measured by the sensor MMA8451Q.
        8th column is the acceleration data in the Y axis measured by the sensor MMA8451Q.
        9th column is the acceleration data in the Z axis measured by the sensor MMA8451Q.
        
        Data are in bits with the following characteristics:

        ADXL345:
        Resolution: 13 bits
        Range: +-16g

        ITG3200
        Resolution: 16 bits
        Range: +-2000°/s

        MMA8451Q:
        Resolution: 14 bits
        Range: +-8g
        '''

        '''
        ADL = 0, Fall = 1
        '''
        self.adl_activities = [f'D{str(i).zfill(2)}' for i in range(1, 20)]  # D01-D19
        self.fall_activities = [f'F{str(i).zfill(2)}' for i in range(1, 16)]  # F01-F15
        
        # ADL for elderly (ONLY ADL, no FALL)
        self.restricted_adl = ['D06', 'D13', 'D18', 'D19']
        
        if not os.path.exists(self.datapath):
            raise ValueError(f"Dataset path not found: {self.datapath}")
            
        self.train_data, self.train_label = self._read_data(train_split)
        self.val_data, self.val_label = self._read_data(val_split)
        self.test_data, self.test_label = self._read_data(test_split)

    def _read_raw_file(self, filepath) : 
        ''' 
            !!!!!CAUTION!!!!!
            I did reference the file that named by "sisfalldataset.ipynb" at directory here.
            https://www.kaggle.com/code/pranavmoothedath4/sisfalldataset
        '''
        try:
            with open(filepath, 'r') as file:
                content = file.read()
            
            content = content.replace(' ', '') # delete blank
            rows = []
            
            for line in content.split(';\n'):
                if line.strip() :   # blank line?
                    try:
                        # ADXL345 acc + ITG3200 gyro == [][:6]
                        values = [float(x) for x in line.split(',')[:6]]
                        rows.append(values)
                    except (ValueError, IndexError):
                        continue
                        
            return np.array(rows)
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            return None

    def _convert_sensor_data(self, data):
        '''
        Convert sensor data from bits to actual units
        
        ADXL345 : 13 bit -> g (+-16g)
        ITG3200 : 16 bit -> °/s (+-2000°/s)
        
        Acceleration [g]: [(2*Range)/(2^Resolution)]*AD
        Angular velocity [°/s]: [(2*Range)/(2^Resolution)]*RD
        '''
        acc_data = data[:, :3] * (2 * 16.0) / (2**13)  # ADXL345
        gyro_data = data[:, 3:6] * (2 * 2000.0) / (2**16)  # ITG3200
        return np.concatenate([acc_data, gyro_data], axis=1)

    def _process_subject_files(self, subject_path, activity_prefix):
        all_data = []
        
        pattern = os.path.join(subject_path, f"{activity_prefix}*.txt")
        activity_files = glob.glob(pattern)
        
        for file_path in activity_files :
            raw_data = self._read_raw_file(file_path)
            
            if raw_data is not None and raw_data.shape[1] == 6 :
                converted_data = self._convert_sensor_data(raw_data)
                all_data.append(converted_data)
        
        if all_data :     # <-------  not empty list
            combined_data = np.concatenate(all_data, axis=0) # all combine
            
            # Removed downsampling
            return combined_data
        
        return None

    def _read_data(self, subject_list) :
        data = []
        labels = []
        
        for subject in subject_list:
            print(f"Processing subject: {subject}")
            subject_path = os.path.join(self.datapath, subject)
            
            if not os.path.exists(subject_path):
                print(f"Subject folder not found: {subject_path}")
                continue
            
            is_elderly = subject.startswith('SE') 
            
            if not (is_elderly and subject != 'SE06'): # you old? but not SE06? (SE06 )
                adl_data = self._process_subject_files(subject_path, 'D')
                if adl_data is not None:
                    filtered_adl = butterworth_filter(adl_data, self.sampling_rate)
                    adl_labels = np.zeros(len(filtered_adl)) # <--------- neccessary??
                    
                    windows_data, windows_labels = self.split_windows(filtered_adl, adl_labels) # split_sequences()
                    if windows_data is not None and len(windows_data) > 0:
                        data.append(windows_data)
                        labels.append(windows_labels)
            
            if not is_elderly or subject == 'SE06':
                fall_data = self._process_subject_files(subject_path, 'F')
                if fall_data is not None:
                    filtered_fall = butterworth_filter(fall_data, self.sampling_rate)
                    fall_labels = np.ones(len(filtered_fall))
                    
                    windows_data, windows_labels = self.split_windows(filtered_fall, fall_labels) # split_sequences
                    if windows_data is not None and len(windows_data) > 0:
                        data.append(windows_data)
                        labels.append(windows_labels)
    
        if len(data) == 0 : 
            # if num of data records is NULL here
            raise ValueError(f"No valid data was loaded for subjects {subject_list}")
            
        final_data = np.concatenate(data, axis=0)
        final_labels = np.concatenate(labels)
        
        unique_labels, counts = np.unique(final_labels, return_counts=True)
        print ()
        print (f"Data distribution for subjects {subject_list}:")
        print (f"Label counts: {list(zip(unique_labels, counts))}\n")
        
        return final_data, final_labels

    def dataset_verbose(self):
        print ()
        print (f"# train: {len(self.train_data)}")
        n_train = dict(Counter(self.train_label))
        print (f"ADL (0): {n_train.get(0, 0)}, Fall (1): {n_train.get(1, 0)}")
        
        print ()
        print (f"# val: {len(self.val_data)}")
        n_val = dict(Counter(self.val_label))
        print (f"ADL (0): {n_val.get(0, 0)}, Fall (1): {n_val.get(1, 0)}")
        
        print ()
        print (f"# test: {len(self.test_data)}")
        n_test = dict(Counter(self.test_label))
        print (f"ADL (0): {n_test.get(0, 0)}, Fall (1): {n_test.get(1, 0)}")

if __name__ == "__main__" :
    sisfall = SisFall()
    sisfall.dataset_verbose()
    sisfall.save_split()