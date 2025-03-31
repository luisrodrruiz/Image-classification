import os
import torch
import sys
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import pandas as pd
import os



# Dataset for  fashion_mnist 
class ImageDataset(Dataset):
    """ Image dataset
        The dataset is expected to be in a csv file, where each row will
        contain at least two columns: filename and label.
    """
    def __init__(self, datafile,  image_path = '', label_map = None):
        """ The constructor loads the dataset into memory
            Parameters:
               datafile: csv file containing the dataset
               filename_col: name of the column in the csv file for the 
                          image files
               label_col: name of the column in the csv file for the labels
               images_path: base path for the image files. If not specified
                            it will simply use the paths in the csv files 
        """
        self.features = []
        self.labels = []
        if not label_map:
            self.label_map = {}
        else:
            self.label_map = label_map
            
        df = pd.read_csv(datafile)
        for index, row in df.iterrows():
            label = row['label']
            if not label in self.label_map:
                self.label_map[label] = len(self.label_map)
            self.labels.append(self.label_map[label])
            filename = row['filename']
            image = np.asarray(Image.open(os.path.join(image_path,filename)),dtype=np.float32)
            self.features.append(np.transpose(image,(2,0,1)))
           
           


        
    def __len__(self):
        return len(self.features)
    
    
    def __getitem__(self, idx):
        features = self.features[idx]                
        label = self.labels[idx]
        return features, label

    def get_sample_shape(self):
        return self.features[0].shape

    def get_num_classes(self):
        return len(self.label_map)



    
