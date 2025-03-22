import torch
import torch.nn as nn
from dataset import ImageDataset
from torch.utils.data import DataLoader
from models import CNNVisionModel
import sys
import os
import argparse


class VisionModelTrainer:
    def __init__(self, train_file, dev_file, image_path = '', batch_size = 16, learning_rate = 0.0001):
        self.model = CNNVisionModel()
        train_dataset = ImageDataset(train_file, image_path = image_path)
        input_foo_sample = torch.zeros(train_dataset.get_sample_shape()).unsqueeze(0)
        print('input_foo_sample = ', input_foo_sample.shape)
        
        out = self.model(input_foo_sample)
        linear_layer_size = out.shape[1]
        n_classes = train_dataset.get_num_classes()
        self.model.add_linear_layer(linear_layer_size,n_classes)
        self.model.cuda()
        print('model = ', self.model)
        self.train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        dev_dataset = ImageDataset(train_file, image_path = image_path)
        self.dev_dataloader = DataLoader(dev_dataset, batch_size = 32, shuffle = False)        
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = learning_rate)

    def train_epoch(self,epoch):
        train_loss = 0.0
        dev_loss = 0.0
        train_accuracy = 0.0
        dev_accuracy = 0.0
        criterion = torch.nn.CrossEntropyLoss()

        self.model.train()
        total_train_samples = 0 
        for batch in self.train_dataloader:
            
            features, labels = batch
            features = features.cuda()
            labels = labels.type(torch.LongTensor).cuda()
            out = self.model(features)
            loss = criterion(out,labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.detach().cpu().item()
            predictions = torch.argmax(out, dim = 1)
            train_accuracy += torch.sum(torch.eq(predictions,labels))
            total_train_samples += len(features)


        train_loss /= total_train_samples
        train_accuracy /= total_train_samples

        self.model.eval()
        total_dev_samples = 0
        for batch in self.dev_dataloader:
            features, labels = batch
            features = features.cuda()
            labels = labels.type(torch.LongTensor).cuda()
            out = self.model(features)
            loss = criterion(out,labels)
            dev_loss += loss.detach().cpu().item()
            predictions = torch.argmax(out, dim = 1)
            dev_accuracy += torch.sum(torch.eq(predictions,labels))
            total_dev_samples += len(features)


        dev_loss /= total_dev_samples
        dev_accuracy /= total_dev_samples
#        print('train_accuracy = ', train_accuracy.item())
        print('EPOCH %d. Train: loss = %5.5f accuracy = %5.1f.  Test:  loss = %5.5f accuracy = %5.1f' %(epoch, train_loss, 100*train_accuracy, dev_loss, 100*dev_accuracy))
            
            
    def train(self,n_epochs = 80):
        for epoch in range(n_epochs):
            self.train_epoch(epoch)
        
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv_file', help = 'csv file for the training dataset. Each row in the csv must have at least two columns: "filename" and "label:', required = True)
    parser.add_argument('--dev_csv_file', help = 'csv file for the dev dataset. Each row in the csv must have at least two columns: "filename" and "label:', required = True)
    parser.add_argument('--image_path', help = 'path that will be prepend to the file names in both the train and dev csv files', required = False, default = '')
    args = parser.parse_args()
    trainer = VisionModelTrainer(args.train_csv_file, args.dev_csv_file, image_path = args.image_path)
    trainer.train()
 
    
if __name__ == "__main__":
    main()
