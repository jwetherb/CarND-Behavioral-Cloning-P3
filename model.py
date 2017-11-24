##############################################################
# Training pipeline for the CarND Behavioral Cloning project #
##############################################################

# Load the 'driving_log.csv' file listing sample snapshots and steering data, etc. from the simulator 
import csv
samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
print('Num supplied samples =', len(samples))

## Split the sample data into training and validation sets. Do this now so we can augment only the training set.
from sklearn.model_selection import train_test_split
train_samples, valid_samples = train_test_split(samples, test_size=0.2)

# Define a generator to feed sample data in batches, to avoid loading the entire sample set into memory
import sklearn
import numpy as np
from scipy.misc import imread
import cv2

def generator(samples, generate_training_data, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            correction = 0.2
            
            for batch_sample in batch_samples:
                
                angle = float(batch_sample[3])

                if (generate_training_data):
                    # For each sample, generate 6 images
                    for i in range(3):
                        source_path = batch_sample[i]
                        tokens = source_path.split('/')
                        local_path = 'data/' + tokens[-2].strip() + '/' + tokens[-1]
                        image = imread(local_path)
                        images.append(image)
                        # Add the reverse of this image
                        flipped_image = cv2.flip(image, 1)
                        images.append(flipped_image)

                    # Add the angles corresponding to the images above
                    
                    # Center
                    angles.append(angle)
                    angles.append(angle * -1)
                    # Left
                    angles.append(angle + correction)
                    angles.append((angle + correction) * -1)
                    # Right
                    angles.append(angle - correction)
                    angles.append((angle - correction) * -1)
                else:
                    # Just add the center image, since this is all the simulator will "see"
                    source_path = batch_sample[0]
                    tokens = source_path.split('/')
                    local_path = 'data/' + tokens[-2].strip() + '/' + tokens[-1]
                    image = imread(local_path)
                    images.append(image)
                    angles.append(angle)

                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)

# Create entry points to compile and train the model using the generator function
training_generator = generator(train_samples, True)
validation_generator = generator(valid_samples, False)

from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

# Define the model
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(16, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(training_generator, 
    samples_per_epoch=len(train_samples), 
    validation_data=validation_generator, 
    nb_val_samples=len(valid_samples), 
    nb_epoch=1)
    
model.save('model.h5')
