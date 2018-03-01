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
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


---
### Files & Command to Execute

#### 1. Project files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* P3-writeup.md summarizing the results

#### 2. To run the code

To train the end-to-end driving CNN model, run
```sh
python model.py
```
The optimized CNN model is then saved into a Checkpoint file "model.hdf5".

Using the [Udacity car simulator](https://github.com/udacity/self-driving-car-sim"), the simulated car can be drive autonomously using:
```sh
python drive.py model.hdf5
```
Note that the CNN model only generates steering angle, the throttle command is generated with a PI controller that regulates the speed at a fixed speed.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The CNN model is adapted from [NVIDIA's self driving car paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 color image   					| 
| Lambda                        | convert color to grayscale, outputs 160x320x1  |
| Cropping2D                    | cropping the image to color image, outputs 80x320x1  |
| Lambda                        | normalize the pixel data to [-1, 1], outputs 80x320x1  |
| Convolution2D 5x5     	| 1x2 stride, valid padding, outputs 76x158x24 	|
| RELU				| introduce nonlinearity	    				|
| Convolution2D 5x5     	| 2x2 stride, valid padding, outputs 36x77x36 	|
| RELU				| introduce nonlinearity	    				|
| Convolution2D 5x5     	| 2x2 stride, valid padding, outputs 16x37x48 	|
| RELU				| introduce nonlinearity	    				|
| Convolution2D 3x3     	| 1x1 stride, valid padding, outputs 14x35x64 	|
| RELU				| introduce nonlinearity	    				|
| Convolution2D 3x3     	| 1x1 stride, valid padding, outputs 12x33x96 	|
| RELU				| introduce nonlinearity	    				|
| Flatten                       | output 38016                                    |
| Dropout                       | keep probability = 0.8                        |
| Fully connected		| output 100        									|
| RELU				| introduce nonlinearity	    				|
| Fully connected		| output 50        									|
| Fully connected               | output 10                                            |
| RELU				| introduce nonlinearity	    				|
| Fully connected               | output 1                                            |


#### 2. Attempts to reduce overfitting in the model

Overfitting of the model is reduced in two ways: 

* The collected data is shuffled and split into training and validation data sets. In the end the optimized CNN model is tested by running on the car simulator. 

* A dropout layer is inserted in the model to make the network fault tolerant and reduce overfitting.

* Both counter clockwise and clockwise driving data are included to avoid overfitting the steering angle to one side.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. The epoch number is tuned such that the validation loss is minimized without overfitting. The dropout layer drop rate is tuned to further improve the validation accuracy and reduce overfitting.

#### 4. Appropriate training data

The training data are generated to support CNN model training:

* Both counter clockwise and clockwise driving data are collected to avoid overfitting steering angle to one side. Three laps of counter clockwise driving data and one lap of clockwise driving data are included in the data set.

* The center, left, and right camera images are all used in the training. The left and right camera data are rectified by adding a biase into the steering angle, such that the car learns how to recover after running off the road.

* The images are mirrored with a negative steering angle to increase the quantity of training data.



For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
