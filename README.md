## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The Project
---
The steps of this project are the following:
* Load the data set into train, test and validate splits
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images


Administrative Stuff
---
1. Instead of Tensorflow, I have used Keras to build the model. The reasons behind this are as follows:
- With some prior experience in Keras, I felt it would be easier to build models using Keras
- Keras uses Tensorflow as its backend, which means all the computation is still taken care of by Tensorflow
- Finally, since there were no requirements mentioning the use of Tensorflow, Keras was the best option. Confirmed with community manager at Udacity
2. The code i.e. jupyter notebook is named [Traffic_Sign_Classifier.ipynb](https://github.com/vikramriyer/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)
3. The html report is named [Traffic_Sign_Classifier.html](https://github.com/vikramriyer/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html)
4. The images downloaded from the web are kept in [external_images](https://github.com/vikramriyer/CarND-Traffic-Sign-Classifier-Project/tree/master/external_images) directory

## Steps

1. Dataset summary

Below are the statistics of the dataset.

|  Total Images  |Total Classes|
|----------|----------|
|51839|43|


|  Data  |Number of images|Percentage|
|----------|-----------|-----------
|Train|34799|67%|
|Validation|4410|8%|
|Test|12630|12630|24%|

The image shape is __32x32x3__ but running models on color images is costly and has no significant benefits in terms of results.
So, in the preprocessing part of the notebook, we convert them to grayscale and hence the new image size is __32x32x1__ where __"1"__ signifies the depth of the image. 

|  Image  | Shape | Channels |
|----------|-----------|-----------
|Original Image|32x32x3|3 (RGB)|
|Processed Image|32x32x1|1 (GRAYSCALE)|


2. Dataset exploration

3. Preprocessing Data

4. Model Architecture

5. Performance on Train, Validation and Test set

6. Performance on Images downloaded from the web

7. Top probabilities using Softmax


Discussion
---

#### Potential Shortcomings in the Project

#### Possible Improvements
