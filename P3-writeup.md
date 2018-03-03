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

[image1]: ./examples/example.jpg "example"
[image2]: ./examples/gray.jpg "Grayscaling"
[image3]: ./examples/crop.jpg "crop"
[image4]: ./examples/flip.jpg "flip"
[image5]: ./examples/center.jpg "center"
[image6]: ./examples/left.jpg "left"
[image7]: ./examples/right.jpg "right"
[image8]: ./examples/curve_road.png "curve"
[image9]: ./examples/CNN_NVIDIA.png "CNN"


---
### Files & Command to Execute

#### 1. Project files

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.hdf5 containing a trained convolution neural network 
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

The CNN model is adapted from [NVIDIA's self driving car paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). Additional Dropout layer is added to reduce overfitting, while multiple RELU layer are added to introduce nonlinearity. The convolution layers are adjusted to get closer to square outputs at each layer (note original images width:height = 4:1).

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
| Convolution2D 3x3     	| 1x1 stride, valid padding, outputs 12x33x64 	|
| RELU				| introduce nonlinearity	    				|
| Flatten                       | output 25344                                    |
| Dropout                       | keep probability = 0.8                        |
| Fully connected		| output 180        									|
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

* Both counter clockwise and clockwise driving data are collected to avoid overfitting steering angle to one side. Two laps of counter clockwise driving data and one lap of clockwise driving data are included in the data set.

* The center, left, and right camera images are all used in the training. The left and right camera data are rectified by adding a biase into the steering angle, such that the car learns how to recover after running off the road.

* Additional data are collected at the curvy parts of the road, so that the car learns to steer harder and not run off the road at the curves.

* The images are mirrored with a negative steering angle to increase the quantity of training data.



For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I took an iterative approach to improve the design of the CNN model. 

I started with the LeNet model for traffic sign classification and removed the final softmax function to get steering angles. The car can follow straight lanes reasonably well, but frequently ran of the road at curves in the simulator. After playing with the model parameters for a long time, it was decided that the LeNet model is not rich enough for self-driving purpose.

Then I switched to [NVIDIA's end-to-end self driving car CNN model](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). This model has significantly more layers and parameters to tune. With the initial dataset and some simple data augmentation techniques, the autonomous car can finish half of the lap. To increase the accuracy of this model, I increased the number of neurons in some layers and inserted several relu layers to incorporate some nonlinearity. The overfitting problem is handled by adding a dropout layer. The final model is a improved CNN model based on NVIDIA's CNN model.

#### 2. Final Model Architecture

The final CNN model is consist of 5 convolutional layers, 4 fully connected layers, 2 Lambda layers, and several Relu and dropout layers. The finial output is a steering angle to the autonomous car.

A illustration of the final CNN model is shown below.
![CNN][image9]

#### 3. Creation of the Training Set & Training Process
Besides creating an approriate CNN model, a large part of effort is devoted to generate good training data for autonomous driving. Since the learned model can only do as good as the training data, producing excellent training data is cruicial to the success of the project.

Two laps of counter clockwise driving data are recored as the initial training data. It is found that color information does not contribute too much to the training accuracy. So the training image are converted into grayscale image.

![example][image1]
![gray][image2]

Since the top part and bottom part of the figure are irrelevant to the driving behavior, these parts are cropped from the original. In addition, the image is mirrored (with reverse steering angle) to provide more training data.

![crop][image3]
![flip][image4]

To teach the car how to recover from off road conditions, the left and right cameras' data are also included into the data set. It is determined that a shift of 0.3 radiance should be added to the steering angle of the center image to get appropriate recovery behavior.
![left][image6]
![right][image7]


It is found that still run off at curve road
![off road][image5]

curve road, revere lap
![curve][image8]


After combining all collected training data (2 forward laps, 1 reverse laps, curvy parts, left/right cameras) and preprocessing (cropping, mirroring, grayscale), I feed the data into keras CNN training function. With all these techniques, there are a total of 66240 training datesets. A random part of the training is splitted as validation data. The final run on the car simulator is considered as the testing data.  

The Adam optimizer is picked to automatically determine model learning rates. The Epoch number is determined to make sure that the validation loss is lowest without overfitting the training data set. To avoid excessive memory usage, generator function is adopted for loading training data into batches. The training is performed on the GPU instance of Amazon Elastic Computing Cloud.

The final training result is shown below:

Epoch 1/5
66048/66240 [============================>.] - ETA: 0s - loss: 0.0263Epoch 00000: val_loss improved from inf to 0.01885, saving model to Model_checkpoints/2018-03-02-model-weights.hdf5
66240/66240 [==============================] - 248s - loss: 0.0263 - val_loss: 0.0189

Epoch 2/5
66048/66240 [============================>.] - ETA: 0s - loss: 0.0195Epoch 00001: val_loss improved from 0.01885 to 0.01813, saving model to Model_checkpoints/2018-03-02-model-weights.hdf5
66240/66240 [==============================] - 244s - loss: 0.0195 - val_loss: 0.0181

Epoch 3/5
66048/66240 [============================>.] - ETA: 0s - loss: 0.0179Epoch 00002: val_loss improved from 0.01813 to 0.01680, saving model to Model_checkpoints/2018-03-02-model-weights.hdf5
66240/66240 [==============================] - 244s - loss: 0.0179 - val_loss: 0.0168

Epoch 4/5
66048/66240 [============================>.] - ETA: 0s - loss: 0.0161Epoch 00003: val_loss improved from 0.01680 to 0.01637, saving model to Model_checkpoints/2018-03-02-model-weights.hdf5
66240/66240 [==============================] - 244s - loss: 0.0161 - val_loss: 0.0164

Epoch 5/5
66048/66240 [============================>.] - ETA: 0s - loss: 0.0146Epoch 00004: val_loss improved from 0.01637 to 0.01597, saving model to Model_checkpoints/2018-03-02-model-weights.hdf5
66240/66240 [==============================] - 243s - loss: 0.0146 - val_loss: 0.0160





#### 4. Discussion for Possible Improvements

We have achieved simple end-to-end learning based self driving in this project. With the current problem setup, there are still lots of places for improvement.

* Associate the steering angle data with current speed of the car. Intuitively, we steer the car much less when driving on the highway with high speed than driving on the city road with low speed. This means that the steering angle output of the CNN model should be less aggressive when the speed is high. We might want to have a two output (steering angle and throttle) CNN model to deal with this problem.

* Develp an analytical model for using training images from left and right cameras. Currently, a heuristic biase constant is added/substracted from the left/right image steering angle. During the develpment of the CNN model, it is observed that left/right images are critical for the car to recover from off lane situations. Thus, it is important to develop a mathmatically sound formula f(camera_dist, car_speed) to calculate the steering angle biase value for left/right camera iamges. 

* Adjust the distribution of training data based on road curvature. While collecting driving data, we can add a parameter for road curvature. Currently, lots of data frames are for straight road driving, which leads to overfitting in zero steering situation. By ploting the road curvature distribution, we can remove some straight driving data or augment curve driving data.

End-to-end learning provides a great alternative for controlling self-driving cars. It requires minimal knowledge about the physics of the car and the environment. However, it is still not clear to me how this method can deal with lane changing, dynamical obstacles, and other more complex scenarios.
