# **Behavioral Cloning** 



---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/forward.jpg "Forward"
[image3]: ./examples/reverse.png "Reverse"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* Nvidia20e_512\Nvidia20e_512.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py Nvidia20e_512\Nvidia20e_512.h5
```

#### 3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model uses the well-known Nvidia CNN architecture for end to end learning for self-driving cars which can be found [in this paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) with slight modifications to match the application.

The model includes ELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model contains two dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, however, the learning rate was manually set to 0.001.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I have used the sample data provided by Udacity, in addition to 3 recorded laps in the forward direction of track 1, 3 recorded laps in the reverse direction of track 1, and specific curve driving recordings in both directions.

### Architecture and Training Documentation

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to test different well know models such as LeNet, Nvidia, VGG and others.

My first step was to use a convolution neural network model as simple as 1 convulution layer and 1 fully connected layer just to make ensure the whole pipeline is intact. 

Afterwards, I have tried LeNet, which was perfectly robust and smooth in straight lines and slight curves, but failed to predict the correct angle in sharp curves (as the one after the bridge).

To introduce more complexity in the model, I have implemented the Nvidia architecture. In order to gauge how well the model was working, I split my image and steering angle data into a training set (80% )and validation set (20%). I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 


To combat the overfitting, I modified the model by adding two dropout layers with keep probability 0.5. The first dropout was after the flatten layer, and the second was after the first fully connected layer. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, such as the same curve after the bridge where the car drove with the right wheels over the curb. To improve the driving behavior in these cases, I have removed the last convulution layer from the Nvidia model, and that was purely based on intuition.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road and at full throttle.

#### 2. Final Model Architecture

The final model architecture (clone.py) consisted of a convolution neural network with the following layers and layer sizes:

|Layer (type) | Output Shape | Param # | Connected to|
|-------------|--------------|---------|-------------|
|image_normalization (Lambda)|(None, 64, 64, 3)|0|lambda_input_1[0][0]|
|convolution_1 (Convolution2D)|(None, 30, 30, 24)|1824|image_normalization[0][0]|
|elu_1 (ELU)|(None, 30, 30, 24)|0|convolution_1[0][0]|
|convolution_2 (Convolution2D)|(None, 13, 13, 36)|21636|elu_1[0][0]|
|elu_2 (ELU)|(None, 13, 13, 36)|0|convolution_2[0][0]|
|convolution_3 (Convolution2D)|(None, 5, 5, 48)|43248|elu_2[0][0]|
|elu_3 (ELU)|(None, 5, 5, 48)|0|convolution_3[0][0]|
|convolution_4 (Convolution2D)|(None, 3, 3, 64)|27712|elu_3[0][0]|
|elu_4 (ELU)|(None, 3, 3, 64)|0|convolution_4[0][0]|
|flatten_1 (Flatten)|(None, 576)|0|elu_4[0][0]|
|dropout_1 (Dropout)|(None, 576)|0|flatten_1[0][0]|
|hidden1 (Dense)|(None, 100)|57700|dropout_1[0][0]|
|elu_5 (ELU)|(None, 100)|0|hidden1[0][0]|
|dropout_2 (Dropout)|(None, 100)|0|elu_5[0][0]|
|hidden2 (Dense)|(None, 50)|5050|dropout_2[0][0]|
|elu_6 (ELU)|(None, 50)|0|hidden2[0][0]|
|hidden3 (Dense)|(None, 10)|510|elu_6[0][0]|
|elu_7 (ELU)|(None, 10)|0|hidden3[0][0]|
|steering_angle (Dense)|(None, 1)|11|elu_7[0][0]|

Total params: 157,691
Trainable params: 157,691
Non-trainable params: 0

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one in the forward direction. Here is an example image of center lane driving:

![alt text][image2]

I then recorded three laps on track one in the reverse direction to further generalize the dataset.

![alt text][image3]

To augment the data sat, I also flipped images and angles thinking that this would give a wider range of examples for the model to train, and also reduce overfitting since such scenes do not exist on track 1.

I have also used the left and right camera images to train the model. I have added the right camera image with a -0.22 offset in the steering angle, and the left camera image with a 0.22 offset in the steering angle.
![alt text][image6]
![alt text][image7]

After the collection process, I had 46,000 data points. I then preprocessed this data by cropping 40 pixels off the top of the images to remove unnecessary details, and 20 pixels off the buttom to remove the vehicle hood. Afterwards, I resized the image to 64x64 to improve performance with square convulution kernels.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20.
