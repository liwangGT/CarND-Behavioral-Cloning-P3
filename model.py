import os
import csv
import datetime

samples = []
with open('Sample_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        # skip the first line, which is the title line
        if line[0] != 'center':
            samples.append(line)
        # for testing implementation only, need to be removed
        if len(samples)>100:
            break

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from keras.models import Sequential, Model
from keras.layers import Lambda, Input, Flatten, Dense, Cropping2D, Convolution2D
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# use generator to avoid large memory consumption
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'Sample_data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                #print("The angle is ", batch_sample[3]) 
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                # flip image to augment data
                #image_flipped = np.fliplr(center_image)
                #angle_flipped = -center_angle
                #images.append(image_flipped)
                #angles.append(angle_flipped)
                # use left and right camera to augment data

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#ch, row, col = 3, 80, 320  # Trimmed image format
inp = (160, 320, 3)
oup = (80, 320, 3)

model = Sequential()
# crop top 50 pixels, bottom 30 pixels, left/right 0 pixels
model.add(Cropping2D(cropping=((50,30), (0,0)), input_shape=inp))
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=inp,
        output_shape=oup))
model.add(Convolution2D(24,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


# model checkpoint
now = datetime.datetime.now()
datenow = now.strftime("%Y-%m-%d-")
#file_path_model = "Model_checkpoints/" + datenow + "model-weights-{epoch:02d}-{val_loss:0.2f}.hdf5"
file_path_model = "Model_checkpoints/" + datenow + "model-weights.hdf5"
checkpoint = ModelCheckpoint(file_path_model, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= \
            len(train_samples), validation_data=validation_generator, \
            nb_val_samples=len(validation_samples), nb_epoch=5, \
            callbacks = callbacks_list, verbose=1)
#history_object = model.fit_generator(train_generator, steps_per_epoch= len(train_samples), \
#                 validation_data=validation_generator, validation_steps=len(validation_samples), \
#                 callbacks = callbacks_list, epochs=5, verbose = 1)


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

