import os
import csv
import datetime
import cv2
import numpy as np
import sklearn
from keras.models import Sequential, Model
from keras.layers import Lambda, Input, Flatten, Dense, Cropping2D, Convolution2D, Dropout
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

def color2gray(xin):
    """
    convert color image to gray
    """
    return (0.21*xin[:,:,:1] +0.72*xin[:,:,1:2]+0.07*xin[:,:,-1:])

def genModel():
    """
    generate CNN model (based on NVIDIA self driving car)
    """
    inp = (160, 320, 3) # initial image size
    oup = (160, 320, 1) # cropped image size

    model = Sequential()
    model.add(Lambda(color2gray, input_shape = inp, output_shape= oup))
    # crop top 50 pixels, bottom 30 pixels, left/right 0 pixels
    model.add(Cropping2D(cropping=((50,30), (0,0))))
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape= oup, output_shape= oup))
    model.add(Convolution2D(24,5,5,subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

def Plot_loss(history_object):
    """
    print history data and plot losses
    """    
    ### print the keys contained in the history object
    print(history_object.history.keys())
    print(history_object.history['loss'])
    print(history_object.history['val_loss'])

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()



def generator(samples, batch_size=32):
    """
    use generator to avoid large memory consumption
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                # append center image
                name = 'Sample_data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                correction = 0.3 # shift angle commands
                # append left camera image
                left_angle = center_angle + correction
                lname = 'Sample_data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(lname)
                images.append(left_image)
                angles.append(left_angle)
                
                # append right camera image
                right_angle = center_angle + correction
                rname = 'Sample_data/IMG/'+batch_sample[1].split('/')[-1]
                right_image = cv2.imread(rname)
                images.append(right_image)
                angles.append(right_angle)

            # flip image to augment data
            Nsample = len(angles)
            for i in range(len(angles)):
                images.append(np.fliplr(images[i]))
                angles.append(-angles[i])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def loadRawData():
    """
    load Raw data for steering angle
    """
    samples = []
    with open('Sample_data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            # skip the first line, which is the title line
            if line[0] != 'center':
                samples.append(line)
            # for testing implementation only, commented for GPU training
            #if len(samples)>100:
            #    break
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return train_samples, validation_samples

def trainModel(model, train_raw, validation_raw):
    """
    training data and save model checkpoints
    """
    # compile and train the model using the generator function
    train_generator = generator(train_raw, batch_size=32)
    validation_generator = generator(validation_raw, batch_size=32)

    # model checkpoint
    now = datetime.datetime.now()
    datenow = now.strftime("%Y-%m-%d-")
    #file_path_model = "Model_checkpoints/" + datenow + "model-weights-{epoch:02d}-{val_loss:0.2f}.hdf5"
    file_path_model = "Model_checkpoints/" + datenow + "model-weights.hdf5"
    checkpoint = ModelCheckpoint(file_path_model, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint]
    model.compile(loss='mse', optimizer='adam')
    # left/center/right images, and all flipped
    ntrain = len(train_raw)*3*2
    nvalid = len(validation_raw)*3*2
    history_object = model.fit_generator(train_generator, samples_per_epoch= \
                ntrain, validation_data=validation_generator, \
                nb_val_samples=nvalid, nb_epoch=10, \
                callbacks = callbacks_list, verbose=1)
    #history_object = model.fit_generator(train_generator, steps_per_epoch= ntrain, \
    #                 validation_data=validation_generator, validation_steps=nvalid, \
    #                 callbacks = callbacks_list, epochs=5, verbose = 1)    
    return history_object


if __name__ == "__main__":
    # load raw training data
    train_raw, validation_raw = loadRawData()

    # generate CNN model
    model = genModel()

    # train CNN model
    history_object = trainModel(model, train_raw, validation_raw)

    # plot training data
    Plot_loss(history_object)
