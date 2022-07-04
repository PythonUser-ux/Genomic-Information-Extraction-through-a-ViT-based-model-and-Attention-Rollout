import sys
import os
import random
import json
import math
import platform
from monai.data import CacheDataset, PersistentDataset


class DatasetCrossVal(CacheDataset):
    def __init__(self, root_dir, split_path, section, num_fold, transforms, seed = 100, val_frac=0.1,
                 cache_num = sys.maxsize, cache_rate=1.0, num_workers=0):    
        if not os.path.isdir(root_dir):
            raise ValueError("Root directory root_dir must be a directory.")
        self.section = section
        
        self.val_frac = val_frac
        
        #self.transforms = transforms
        self.num_fold = num_fold
        self.seed = seed
        
        data = self._generate_data_list(split_path)
        super().__init__(data, transforms, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers)
        
     
    #split data in train, val and test sets in a reproducible way
    def _generate_data_list(self, split_path):
        with open(split_path) as fp:
           mri_path=json.load(fp)
        data = list()
        
        if self.section == 'test':
            data = mri_path[f'fold{self.num_fold}']['test']
        elif self.section == 'training' or self.section == 'validation':
            datalist = mri_path[f'fold{self.num_fold}']['train']
        
            datalist.sort(key=lambda item: item.get('mask'))  # make sure that the datalist have a fixed order before shuffling
            random.seed(self.seed)
            random.shuffle(datalist) # shuffles the ordering of datalist (deterministic given the chosen seed)
            
            val_items = math.ceil(self.val_frac * len(datalist))
            train_items = len(datalist) - val_items 
            assert len(datalist) == (train_items + val_items), "Error dataset split"
            if self.section == 'training':
                data = datalist[:train_items]
            elif self.section == 'validation':
                data = datalist[train_items:]
            
        else: 
            raise ValueError(
                    f"Unsupported section: {self.section}, "
                    "available options are ['training', 'validation', 'test']."
                )
        
        if platform.system() != 'Windows':
            for sample in data:
                for key in sample.keys():
                    sample[key] = sample[key].replace('\\', '/')
        return data     

    
class Dataset(CacheDataset):
    def __init__(self, root_dir, data_path, section, num_fold, transforms, seed = 100, val_frac=0.1, test_frac = 0.1,
                 cache_num = sys.maxsize, cache_rate=1.0, num_workers=0):
        if not os.path.isdir(root_dir):
            raise ValueError("Root directory root_dir must be a directory.")
        self.section = section

        self.val_frac = val_frac
        self.test_frac = test_frac
        self.num_fold = num_fold
        
        #self.transforms = transforms
        self.seed = seed
        
        data = self._generate_data_list(data_path)
        super().__init__(data, transforms, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers)
        
     
    #split data in train, val and test sets in a reproducible way
    def _generate_data_list(self, data_path):
        with open(data_path) as fp:
           paths=json.load(fp)
        data = list()
        if self.section == 'test':
            data = paths[f'fold{self.num_fold}']['test']
        if self.section == 'training' or self.section == 'validation' or self.section == 'test':
            datalist = paths[f'fold{self.num_fold}']['train']
        
            datalist.sort(key=lambda item: item.get('mask'))  # make sure that the datalist have a fixed order before shuffling
            random.seed(self.seed)
            random.shuffle(datalist) # shuffles the ordering of datalist (deterministic given the chosen seed)
            
            val_items = math.ceil(self.val_frac * len(datalist))
            test_items = math.ceil(self.test_frac * len(datalist))
            train_items = len(datalist) - val_items  - test_items
            assert len(datalist) == (train_items + val_items + test_items), "Error dataset split"
            if self.section == 'training':
                data = datalist[:train_items]
            elif self.section == 'validation':
                data = datalist[train_items:train_items+val_items]
            elif self.section == 'test':
                data = datalist[-test_items:]
            
        else: 
            raise ValueError(
                    f"Unsupported section: {self.section}, "
                    "available options are ['training', 'validation', 'test']."
                )
        
        if platform.system() != 'Windows':
            for sample in data:
                for key in sample.keys():
                    sample[key] = sample[key].replace('\\', '/')
        return data  