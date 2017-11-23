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
        
print('Num samples =', len(samples))

# Load the image files (center/left/right) from disk, and adjust the steering direction for the left and right
# images, since they will be taught to the model as if they were center images
from scipy.misc import imread
images = []
directions = []
correction = 0.2
for line in samples:
    # Use all three cameras. For right and left cams, modify the steering direction since it will be taught to the 
    # model as coming from the center cam
    for i in range(3):
        source_path = line[i]
        tokens = source_path.split('/')
        local_path = 'data/' + tokens[-2].strip() + '/' + tokens[-1]
        image = imread(local_path)
        images.append(image)
    direction = float(line[3])
    directions.append(direction)
    directions.append(direction + correction)
    directions.append(direction - correction)
    
print(len(images))
print(len(directions))

import opencv as cv2
augmented_images = []
augmented_directions = []
for image, direction in zip(images, directions):
    augmented_images.append(image)
    augmented_directions.append(direction)
    flipped_image = cv2.flip(image, 1)
    flipped_direction = direction * -1.0
    augmented_images.append(flipped_image)
    augmented_directions.append(flipped_direction)  

## Split the sample data into training and validation sets
#from sklearn.model_selection import train_test_split
#train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Define a generator to feed sample data in batches, to avoid loading the entire sample set into memory
import numpy as np
import sklearn
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Create entry points to compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format

#TODO flip the images and train them in reverse
#image_flipped = np.fliplr(image)
#measurement_flipped = -measurement

# Use all three cameras
#with open('data/driving_log.csv', 'r') as f:
#    reader = csv.reader(f)
#with open('data/driving_log.csv') as csvfile:
#    reader = csv.reader(csvfile)
#    for row in reader:
for row in samples:
    steering_center = float(row[3])
    
    # create adjusted steering measurements for the side camera images
    correction = 0.2 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    # read in images from center, left and right cameras
    path = "data/IMG" # fill in the path to your training IMG directory
    img_center = process_image(np.asarray(Image.open(path + row[0])))
    img_left = process_image(np.asarray(Image.open(path + row[1])))
    img_right = process_image(np.asarray(Image.open(path + row[2])))

    # add images and angles to data set
    car_images.extend(img_center, img_left, img_right)
    steering_angles.extend(steering_center, steering_left, steering_right)

from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Define the model
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(ch, row, col)))
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

model.fit_generator(train_generator, 
    samples_per_epoch=len(train_samples), 
    validation_data=validation_generator, 
    nb_val_samples=len(validation_samples), 
    nb_epoch=3)
    
model.save('model.h5')
