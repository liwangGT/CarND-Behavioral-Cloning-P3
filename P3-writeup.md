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

The CNN model is adapted from [NVIDIA's self driving car paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). Additional Dropout layer is added to reduce overfitting, while multiple RELU layer are added to introduce nonlinearity.The convolution layers are adjusted to get closer to square outputs at each layer (note original iamge has).

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

* Both counter clockwise and clockwise driving data are collected to avoid overfitting steering angle to one side. Three laps of counter clockwise driving data and one lap of clockwise driving data are included in the data set.

* The center, left, and right camera images are all used in the training. The left and right camera data are rectified by adding a biase into the steering angle, such that the car learns how to recover after running off the road.

* Additional data are collected at the curvy parts of the road, so that the car learns to steer harder and not run off the road at the curves.

* The images are mirrored with a negative steering angle to increase the quantity of training data.



For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I took an iterative approach to improve the design of the CNN model. 

I started with the LeNet model for traffic sign classification and removed the final softmax function to get steering angles. The car can follow straight lanes reasonably well, but frequently ran of the road at curves in the simulator. After playing with the model parameters for a long time, it was decided that the LeNet model is not rich enough for self-driving purpose.

Then I switched to [NVIDIA's end-to-end self driving car CNN model](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). This model has significantly more layers and parameters to tune. With the initial dataset and some simple data augmentation techniques, the .

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

Original image
![example][image1]
Convert to gray
![gray][image2]
crop image
![crop][image3]
flip image
![flip][image4]
center
![center][image5]
left
![left][image6]
right
![right][image7]
curve
![curve][image8]
CNN model
![CNN][image9]

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


#### 4. Discussion for Possible Improvements

We have achieved simple end-to-end learning based self driving in this project. With the current problem setup, there are still lots of places for improvement.

* Associate the steering angle data with current speed of the car. Intuitively, we steer the car much less when driving on the highway with high speed than driving on the city road with low speed. This means that the steering angle output of the CNN model should be less aggressive when the speed is high. We might want to have a two output (steering angle and throttle) CNN model to deal with this problem.

* Develp an analytical model for using training images from left and right cameras. Currently, a heuristic biase constant is added/substracted from the left/right image steering angle. During the develpment of the  f(Dist_camera, speed)

* Adjust the distribution of training data based on road curvature. While collecting driving data, we can add a parameter for road curvature. Currently, lots of data frames are for straight road driving, which leads to overfitting in zero steering situation. By ploting the road curvature distribution, we can remove some straight driving data or augment curve driving data.

End-to-end learning provides a great alternative for controlling self-driving cars. It requires minimal knowledge about the physics of the car and the environment. However, it is still not clear to me how this method can deal with lane changing, dynamical obstacles, and other more complex scenarios.  
