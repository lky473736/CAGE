from dataset import *
from dataset.dataset_generator import HARDataGenerator

import glob
import numpy as np
import pandas as pd
import os
import re

class UMAFall(HARDataGenerator) :
    def __init__(self, window_length=128, clean=False, fall=True):
        super(UMAFall, self).__init__()
        self.clean = clean
        self.fall = fall
        self.sampling_rate = 20  # <---- sampling rate now
        self.WINDOW_LENGTH = window_length
        self.STRIDE = self.WINDOW_LENGTH // 2
        
        self.adl_activities = [
            'Aplausing',
            'HandsUp', 
            'MakingACall',
            'OpeningDoor',
            'Sitting_GettingUpOnAChair',
            'Walking',
            'Bending',
            'Hopping',
            'LyingDown_OnABed',
            'GoDownstairs',
            'GoUpstairs',
            'Jogging'
        ]
        
        self.fall_activities = [
            'forwardFall',
            'backwardFall',
            'lateralFall'
        ]
        
        self.label2id = {act: 0 for act in self.adl_activities}
        self.label2id.update({act: 1 for act in self.fall_activities})
        
        self.datapath = "data/UMAFall_Dataset"

        self.split_data()

    def _parse_filename(self, filename) :
        pattern = r'UMAFall_Subject_(\d+)_(ADL|Fall)_([a-zA-Z_]+)_(\d+)_(\d{4}-\d{2}-\d{2})_.*\.csv' # <----- regex about file name. because of "ADL? FALL?"
        match = re.match(pattern, filename)
        
        if match : # file name has no error?
            subject_id = int(match.group(1))
            activity_type = match.group(2)
            activity = match.group(3)
            trial = int(match.group(4))
            date = match.group(5)
            return subject_id, activity_type, activity, trial, date
        return None

    def _read_sensor_file(self, filepath) :
        data_lines = []
        chest_id = None
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines :
                '''                                
                    %f8:95:c7:f3:ba:82; 0; RIGHTPOCKET; lge-LG-H815-5.1                                
                    %C4:BE:84:71:A5:02; 2; WAIST; SensorTag                                            
                    %C4:BE:84:70:0E:80; 3; WRIST; SensorTag                                            
                    %B0:B4:48:B8:77:03; 4; ANKLE; SensorTag                                            
                    %C4:BE:84:70:64:8A; 1; CHEST; SensorTag                                            
                                                                                                    
                    % Sensor_Type:                                                                     
                    % Accelerometer = 0                                                                
                    % Gyroscope = 1                                                                    
                    % Magnetometer = 2                                      
                '''
                if 'CHEST; SensorTag' in line : # ONLY USE CHEST
                    chest_id = line.split(';')[1].strip()
                    break
                    
            if not chest_id : # no chest? 
                return None
                
            data_start = False
            for line in lines:
                if '% TimeStamp; Sample No;' in line :
                    data_start = True
                    continue
                
                if data_start :
                    try:
                        values = [float(v.strip()) for v in line.split(';')]
                        sensor_id = str(int(values[-1]))
                        sensor_type = int(values[-2])
                        
                        # keep accelerometer (0) and gyroscope (1) data from CHEST
                        if sensor_id == chest_id and sensor_type in [0, 1]:
                            data_lines.append([*values[2:5], sensor_type])  # X,Y,Z,sensor_type
                    except :
                        continue
        return np.array(data_lines) if data_lines else None

    def split_data(self):
        all_files = glob.glob(os.path.join(self.datapath, "*.csv"))
        
        subject_files = {}
        for f in all_files :
            filename = os.path.basename(f)
            info = self._parse_filename(filename)
            if info:
                subject_id = info[0]
                if subject_id not in subject_files :
                    subject_files[subject_id] = []
                subject_files[subject_id].append(f)
        
        train_subjects = list(range(1, 14))              # 1-13
        val_subjects = list(range(14, 17))               # 14-16
        test_subjects = list(range(17, 20))               # 17-19
        
        self.train_files = []
        self.val_files = []
        self.test_files = []
        
        for subject_id, files in subject_files.items() :
            if subject_id in train_subjects :   # train
                self.train_files.extend(files)
            elif subject_id in val_subjects :   # val
                self.val_files.extend(files)
            else :                              # test
                self.test_files.extend(files)
        
        self.train_data, self.train_label = self._process_files(self.train_files)
        self.val_data, self.val_label = self._process_files(self.val_files)
        self.test_data, self.test_label = self._process_files(self.test_files)

    def _process_files(self, files):
        data = []
        labels = []
        
        for filepath in files :
            print (f"Processing file: {filepath}") 
            try:
                filename = os.path.basename(filepath)
                info = self._parse_filename(filename)
                if not info:
                    print(f"Skipping file: {filename}")
                    continue
                    
                _, activity_type, activity, _, _ = info
                
                sensor_data = self._read_sensor_file(filepath)
                if sensor_data is None or len(sensor_data) == 0:
                    print(f"Skipping file: {filename} - No sensor data")
                    continue
                
                acc_data = sensor_data[sensor_data[:, -1] == 0][:, :3]
                gyro_data = sensor_data[sensor_data[:, -1] == 1][:, :3]
                
                if len(acc_data) == 0 or len(gyro_data) == 0:
                    print (f"Skipping file: {filename} - Insufficient acc or gyro data")
                    continue
                    
                merged_data = np.column_stack((acc_data, gyro_data))
                
                if activity in self.fall_activities:
                    label = 1
                else:
                    label = 0

                windows_data, windows_labels = self.split_windows(merged_data, 
                                                                np.full(len(merged_data), label))
                
                if windows_data is not None and len(windows_data) > 0:
                    data.append(windows_data)
                    labels.append(windows_labels)
                    print(f"Successfully processed file: {filename}")
                else:
                    print(f"Skipping file: {filename} - No windows created")
                    
            except Exception as e:
                print(f"Error processing {filepath}: {str(e)}")
                import traceback
                traceback.print_exc() 
                continue
        
        if len(data) == 0:
            print("No valid data was loaded at all!")
            raise ValueError("No valid data was loaded")
        
        final_data = np.concatenate(data, axis=0)
        final_labels = np.concatenate(labels)
        unique_labels, counts = np.unique(final_labels, return_counts=True)
        
        print ()
        print ("Final Data Distribution:")
        
        for label, count in zip(unique_labels, counts) :
            print(f"Label {label}: {count} samples")
        
        return final_data, final_labels

if __name__ == "__main__" :
    umafall = UMAFall()
    umafall.dataset_verbose()
    umafall.save_split()