# Image-classification
This repository implements several deep learning models for image classification. The current available models are: CNN and Vision Transformer (ViT).

The datasets for training the models consists of  image and csv files where the csv file must contain at least two columns "filename" and "label" 

**Requirements**

- pytorch >= 2.0
- pandas
- Pillow 

**Usage**

The main script takes the following parameters:

- train_csv_file: csv file containing the trainining data
- dev_csv_file: csv file containing the development (or validation) data 

Optionally, the following parameters can be specified:

- model: model to use ("cnn" or "vit")
- image_path: path to be prepended to the "filename" column in the csv files
- out_dir: Output directory where the trained models will be saved
