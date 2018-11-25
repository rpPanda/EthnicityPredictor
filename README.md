# Ethnicity Classifier
A simple program using Resnet, Transfer Learning, Metric Learning and KNN-Classifiers to predict the ethnicity of a person using their images. The classification results were based on results from Google Images. Face Recognization API was used to extract facial features and compare them using Metric Learning. KNN-Classifier was used to cluster training dataset and test new images.

For more information about the Face Recognition API : https://github.com/ageitgey/face_recognition

## Requirements

- Python 3.3+
- Linux

### Required libraries and dependencies

```
python pip3 install face_recognition
python pip3 install Pillow
python pip3 install numpy, scipy, scikit-learn
python pip3 install google_images_download
```

## Usage

To Download the dataset run the prep_dataset.py
```
python3 prep_dataset.py
```
It will create a 'train' directory with 100 images of each ethnicity (separate folders for male and female)

To train the model run training.py
```
python3 training.py
```
It shows the progress of the training and stores the weights for prediction

To test the model create a 'test' directory and add images

Currently supported formats include : jpg,jpeg and png

Run the testing.py
```
python3 testing.py
```
