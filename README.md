# Capstone - Age Estimation using Convolutional Neural Networks
![]()  
[Roy Kim](www.roykim.me)

## Executive Summary
The problem statement and goal for this project is as follows: Using convolutional neural networks, I will train a classification model to predict someone’s age group based upon his/her picture. Model performance will be guided by cross entropy and accuracy (as well as considering the confusion matrix and other metrics), and the model will aim to achieve a score greater than baseline accuracy (determined by the number of classes).

The data was retrieved from multiple sources, such as the [MegaAge](http://mmlab.ie.cuhk.edu.hk/projects/MegaAge/) dataset, the [IMDB and Wiki](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) dataset. These images come preprocessed, meaning that the face was already extracted from each image, and the size of each face image was standardized to be 178x218 pixels. However, due to limitations of hardware, only the MegaAge dataset was used.

The metrics that were used in this project were cross entropy (mainly as a loss function for the convolutional neural network) and accuracy. However, in this case of multiclass prediction where classes were not completely independent of each other (ie. the age group of 21-30 should exhibit more common characteristics with the age group of 31-40 than the age group of 71-80), the confusion matrix was also considered.

There were a few models that were trained on the dataset. The first was a model that was trained from scratch on local hardware. This model achieved an accuracy score of about 12%, meaning that it correctly classified the age group of the person in the picture only 12% of the time. Going back to the images, I applied a grayscale on each image to get an improved accuracy score of about 25-40%. (Section on confusion matrix, other metrics)

The second model used a pre-trained model (the [VGG-16 Face model](https://github.com/rcmalli/keras-vggface)) to help with classification. Using this model, the age estimations achieved an accuracy score of about 

Some of the limitations of these findings are due to the nature of 
What risks/limitations/assumptions affect these findings?

## Statistical Analysis

implementation
evaluation
inference

## Process
- ### Data collection and processing:
Data was collected through the links above and hosted locally. After downloading the data, I placed each of the images into a new directory based upon the age group classes they would fall in. When doing this, I realized that there was high class imbalance. As such, I decided to limit the scope of the dataset and only allow 1500 images per class as a maximum in the train dataset, and 500 images per class as a maximum in the test dataset. Furthermore, I originally had 10 classes, ranging from 0-10, 11-20, ..., 81-90, and >90. However, the downloaded dataset did not have anyone in the 71-80, 81-90, and >90 age groups, so those classes were eliminated (for a total of 7 classes remaining). After looking at the normalized value counts, the baseline accuracy for the model was about 22%.

- ### Creating the first model:
The first model was created from the ground up using Keras on my local machine. After checking that the images were placed correctly in the formatted file structure, the `flow_from_directory` method was used to feed the images into the model. I initially set the number of epochs to 5, feeding in all the train and test images (about 7500 and 2300 respectively). The model scored lower than the baseline, with an accuracy score of about 19%. This model was saved as a `.json` file, with the weights saved in a `.h5` file, as well. In fact, the model was not learning anything, just choosing the majority train class (age group '31-40') for all predictions.

- ### Using the pre-trained VGG-16 Faces model:
The second model was created by using the pre-trained VGG-16 Face model. 

- ### Extra: Applying the model to a live video feed via OpenCV
