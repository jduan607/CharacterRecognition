# STAT479 Project
Project for STAT479: Deep Learning (Spring 2019) at University of Wisconsin-Madison

## Background
The recognition and classification of images is a very popular subject in computer science studies and other related fields. In this
project, we will apply deep learning algorithms to develop a method to classify English letters using different types of
images on both the uppercase and lowercase of all 26 English letters. This project attempts to find a classification model that could generalizing well for identifying every English character from different types of images.

## Deep Learning Models
+ Multilayer Perceptron (MLP)
+ ResNet-50
+ Inception-v3

## Software
+ Python for implementing and train our models
+ R for cleaning the dataset and analyzing the output predictions of models.

## Result
After comparing all models, we concluded that the CNN model combining the structure of ResNet-50 and the structure of multilayer perceptron with three hidden layers is the most effective to classify images in our dataset. In contrast, MLP models are not as powerful as the CNN model, although their simpler architectures allow for faster training process. In particular, CNN model is especially effective for classifying handwritten images. In this project, by combining the ResNet-50 architecture with three fully connected layers, using momentum learning and cropping input images, the model we built could generalize well not only to the relatively large number of classes in our dataset but also to different types of images.
