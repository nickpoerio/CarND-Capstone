import os
import cv2
import numpy as np

main_dir = 'site_images'
samples = []
directories = os.listdir(main_dir)
for dir in directories:
    samples+=['/'+dir+'/'+x for x in os.listdir(main_dir+'/'+dir)]

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=8, training=False):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            states = []
            for name in batch_samples:
                image = cv2.imread(main_dir+'/'+name)
                #image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB)
                if name[1:4]=='red':
                    state = 1
                else:
                    state = 0
                    
                images.append(image)
                states.append(state)
                if training==True:
                    # Flipping image
                    images.append(cv2.flip(image,1))
                    states.append(state)
                    '''# image brightness 1
                    images.append(np.where((255 - image) < 30,255,image+30))
                    states.append(state)
                    # image brightness 2
                    images.append(np.where(image < 30,0,image-30))
                    states.append(state)'''
                    
            # trim image to only see section with road
            X = np.array(images)
            y = np.array(states)
            yield shuffle(X, y)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=8, training=True)
validation_generator = generator(validation_samples, batch_size=8, training=False)

from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Conv2D, Dropout, Flatten, Dense, MaxPooling2D
from keras.regularizers import l2
from keras import metrics

def Preprocessing():
    model = Sequential()
    # aggiungere subsampling
    model.add(Lambda(lambda x: (x/255.)-.5, input_shape=(600,800,3)))
    model.add(Cropping2D(cropping=((100,100),(0,0))))
    model.add(MaxPooling2D(pool_size=2))
    return model

def NVIDIAmodel(drop_rate1=0.,drop_rate2=0.):
    reg_rate = .01
    model = Preprocessing()
    model.add(Conv2D(24,5, strides=(2,2), activation='relu'))
    model.add(Dropout(drop_rate1))
    model.add(Conv2D(36,5, strides=(2,2), activation='relu'))
    model.add(Dropout(drop_rate1))
    model.add(Conv2D(48,5, strides=(2,2), activation='relu'))
    model.add(Dropout(drop_rate1))
    model.add(Conv2D(64,3, activation='relu'))
    model.add(Dropout(drop_rate1))
    model.add(Conv2D(64,3, activation='relu'))
    model.add(Dropout(drop_rate1))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(drop_rate2))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(drop_rate2))
    model.add(Dense(10, activation='relu',kernel_regularizer=l2(reg_rate),bias_regularizer=l2(reg_rate)))
    model.add(Dense(1,activation='sigmoid',kernel_regularizer=l2(reg_rate),bias_regularizer=l2(reg_rate)))
    return model

model = NVIDIAmodel(drop_rate1=0.1,drop_rate2 = 0.33)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[metrics.binary_accuracy])

history_object = model.fit_generator(train_generator, steps_per_epoch= len(train_samples)/8,
validation_data=validation_generator, validation_steps=len(validation_samples)/8, epochs=8, verbose = 1)

model.save("TL_site_classifier.h5")
model.save_weights("TL_site_classifier_weights.h5")
