import csv
import cv2
import numpy as np

images = []
measurements = []
files = ['data']
# files = ['recorded data/track 1/3/', 'recorded data/track 1/4/',  'recorded data/track 1/corners/']

# files = ['recorded data/track 1/3/', 'recorded data/track 1/4/']

# files = ['recorded data/track 1/corners/']

for f in files:
    lines = []
    with open(f+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    for line in lines[1:]:
        center_path = line[0]
        left_path = line[1]
        right_path = line[2]

        center_filename = (center_path.split('/')[-1]).split('\\')[-1]
        left_filename = (left_path.split('/')[-1]).split('\\')[-1]
        right_filename = (right_path.split('/')[-1]).split('\\')[-1]

        center_current_path = f + '/IMG/' + center_filename
        left_current_path = f + '/IMG/' + left_filename
        right_current_path = f + '/IMG/' + right_filename

        center_image = cv2.imread(center_current_path)
        left_image = cv2.imread(left_current_path)
        right_image = cv2.imread(right_current_path)

        measurement = float(line[3])
        # if abs(measurement) <= 0.01:
        #     continue
        images.append(center_image)
        measurements.append(measurement)

        images.append(left_image)
        measurements.append(measurement + 0.25)

        images.append(right_image)
        measurements.append(measurement - 0.25)

X_train = np.array(images)
y_train = np.array(measurements)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Dropout, Cropping2D, Activation
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU


def LeNet():
    LeNet = Sequential()
    LeNet.add(Cropping2D(cropping=((40,20),(0,0)), input_shape=(160,320,3)))
    LeNet.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(100,320,3)))

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
    return LeNet


def nvidia_model(input_shape):
    model = Sequential()
    model.add(Cropping2D(cropping=((60, 25), (0, 0)), input_shape=input_shape))
    model.add(Lambda(lambda x: x / 127.5 - 1.0))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
    model.add(Activation('elu'))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
    model.add(Activation('elu'))

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
    model.add(Activation('elu'))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    model.add(Activation('elu'))

    # model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    # model.add(Activation('elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.5))
    # model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    return model

model = nvidia_model((160,320,3))

adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss='mse')
model.fit(x=X_train, y=y_train, nb_epoch=20, batch_size=128,  validation_split=0.2, shuffle=True)

model.save('LeNet.h5')
"""


model = Sequential()

model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(80,320,3)))

model.add(Convolution2D(24, 5, 5, name="convolution_1", subsample=(2, 2), border_mode="valid", init='he_normal'))
model.add(ELU())
model.add(Convolution2D(36, 5, 5, name="convolution_2", subsample=(2, 2), border_mode="valid", init='he_normal'))
model.add(ELU())
model.add(Convolution2D(48, 5, 5, name="convolution_3", subsample=(2, 2), border_mode="valid", init='he_normal'))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, name="convolution_4", border_mode="valid", init='he_normal'))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, name="convolution_5", border_mode="valid", init='he_normal'))
model.add(ELU())

model.add(Flatten())
model.add(Dropout(0.6))

model.add(Dense(100, name="hidden1", init='he_normal'))
model.add(ELU())
model.add(Dense(50, name="hidden2", init='he_normal'))
model.add(ELU())
model.add(Dense(10, name="hidden3", init='he_normal'))
model.add(ELU())

model.add(Dense(1, name="steering_angle", activation="linear"))
model.summary()
adam = Adam(lr=0.0001)

model.compile(loss='mse', optimizer=adam)
model.fit(x=X_train, y=y_train, nb_epoch=10, validation_split=0.2, shuffle=True)
model.save('nvidia.h5')
"""

