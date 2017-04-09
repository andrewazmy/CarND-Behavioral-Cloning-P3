import csv
import cv2
import numpy as np


lines = []

with open('recorded data/track 1/2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines[1:]:
    source_path = line[0]
    filename = source_path.split('\\')[-1]
    current_path = 'recorded data/track 1/2/IMG/' + filename
    image = cv2.imread(current_path)
    measurement = float(line[3])
    if abs(measurement) <= 0.01:
        continue
    images.append(image)
    measurements.append(measurement)




lines = []

with open('recorded data/track 1/1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

for line in lines[1:]:
    source_path = line[0]
    filename = source_path.split('\\')[-1]
    current_path = 'recorded data/track 1/1/IMG/' + filename
    image = cv2.imread(current_path)
    measurement = float(line[3])
    if abs(measurement) <= 0.01:
        continue
    images.append(image)
    measurements.append(measurement)


lines = []

with open('recorded data/track 1/0/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

for line in lines[1:]:
    source_path = line[0]
    filename = source_path.split('\\')[-1]
    current_path = 'recorded data/track 1/0/IMG/' + filename
    image = cv2.imread(current_path)
    measurement = float(line[3])
    if abs(measurement) <= 0.01:
        continue
    images.append(image)
    measurements.append(measurement)


lines = []

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

for line in lines[1:]:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.imread(current_path)
    measurement = float(line[3])
    if abs(measurement) <= 0.01:
        continue
    images.append(image)
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Dropout
from keras.regularizers import l2, activity_l2

simple_model = Sequential()
simple_model.add(Flatten(input_shape=(160,320,3)))
simple_model.add(Dense(1))
simple_model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))


LeNet = Sequential()
LeNet.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

LeNet.add(Convolution2D(6, 5, 5, subsample=(1, 1), border_mode='valid', activation='elu'))
LeNet.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), border_mode='valid'))

LeNet.add(Convolution2D(16, 5, 5, subsample=(1, 1), border_mode="valid", activation='elu'))
LeNet.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), border_mode='valid'))

LeNet.add(Flatten())

LeNet.add(Dropout(0.5))
LeNet.add(Dense(120, activation='elu'))

LeNet.add(Dropout(0.5))
LeNet.add(Dense(84, activation='elu'))

LeNet.add(Dense(10, activation='elu'))

LeNet.add(Dense(1))



LeNet.compile(loss='mse', optimizer='adam')
LeNet.fit(x=X_train, y=y_train, nb_epoch=20, validation_split=0.2, shuffle=True)

LeNet.save('LeNet.h5')