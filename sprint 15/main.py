'''facial recognition model to predict a person's age'''

# libraries
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam

from Interfaces.DataLoader import *


# load data
image_path = r'data/faces/'
labels_path = r'data/faces/labels.csv'
labels = DataLoader.from_csv(path)

# EDA
print(labels.head(15))
print(labels['file_name'].describe())
print(labels['real_age'].describe())

labels['file_name'].nunique()
labels['real_age'].hist()
labels.drop('file_name', axis = 1, inplace = True)
print(labels.head())

# view sample images
sample = 20
rng = np.random.default_rng()
random_sample =[int(rng.uniform(low = 0, high = 1000)) for _ in range(0,sample)]
square = 12


fig, axs = plt.subplots((sample + 9) // 10, min(sample, 10), figsize = (square,square))
axs = axs.flatten()

for n, pic in enumerate(random_sample):
    sample_image = Image.open(f'{image_path}{pic:06d}.jpg')
    axs[n].imshow(sample_image)
    axs[n].axis('off')
    plt.tight_layout()
    

sample_image.size

'''
There are 7591 images of different sizes and quality. According to labels.csv, the images are of people ranging from 1 - 100 years old. There is no missing data.
'''

# load train data
def load_train(path, target_size, validation_split,batch_size = 16,  seed = 12345):
    
    """
    It loads the train part of dataset from path
    """
    
    # create instance of Image generate with below parameters
    train_gen = ImageDataGenerator(
    validation_split=validation_split,
        rescale=1./255
        # horizontal_flip=True,
        # vertical_flip=True,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # rotation_range=90
    )
    
    # generator from ImageDataGenerator instance to  load images from a path
    train_gen_flow = train_gen.flow_from_directory(
        path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='training',
        seed=seed
    )

    return train_gen_flow

# load test data
def load_test(path, target_size, validation_split,batch_size = 16,  seed = 12345):
    
    """
    It loads the validation/test part of dataset from path
    """
    
    test_gen = ImageDataGenerator(
    validation_split=validation_split,
        rescale=1./255
        # horizontal_flip=True,
        # vertical_flip=True,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # rotation_range=90
    )
    
    test_gen_flow = test_gen.flow_from_directory(
        path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation',
        seed=seed
    )    

    return test_gen_flow

# Define model (ResNet50)
def create_model(input_shape):  
    backbone = ResNet50(weights='imagenet', 
                        input_shape=input_shape,
                        include_top=False)

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1, activation='linear'))

    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model

# train model
def train_model(model, train_data, test_data, batch_size=None, epochs=20,
                steps_per_epoch=None, validation_steps=None):

    """
    Trains the model given the parameters
    """
    
    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
        
    if validation_steps is None:
        validation_steps = len(test_data)

    model.fit(train_data, 
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2)

    return model

# prepare to run on the GPU platform

init_str = """
import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
"""

import inspect

with open('run_model_on_gpu.py', 'w') as f:
    
    f.write(init_str)
    f.write('\n\n')
        
    for fn_name in [load_train, load_test, create_model, train_model]:
        
        src = inspect.getsource(fn_name)
        f.write(src)
        f.write('\n\n')


'''
Epoch 1/20 356/356 - 35s - loss: 95.3532 - mae: 7.4339 - val_loss: 124.3362 - val_mae: 8.4921 
Epoch 2/20 356/356 - 35s - loss: 76.8372 - mae: 6.6707 - val_loss: 127.6357 - val_mae: 8.6035 
Epoch 3/20 356/356 - 35s - loss: 69.9428 - mae: 6.3992 - val_loss: 91.1531 - val_mae: 7.4454 
Epoch 4/20 356/356 - 35s - loss: 64.4249 - mae: 6.1407 - val_loss: 124.0287 - val_mae: 8.3481 
Epoch 5/20 356/356 - 35s - loss: 52.8486 - mae: 5.5913 - val_loss: 109.1004 - val_mae: 8.2192 
Epoch 6/20 356/356 - 35s - loss: 46.3094 - mae: 5.2223 - val_loss: 85.1038 - val_mae: 7.0332 
Epoch 7/20 356/356 - 35s - loss: 38.2617 - mae: 4.7951 - val_loss: 92.0900 - val_mae: 7.3359 
Epoch 8/20 356/356 - 35s - loss: 37.4804 - mae: 4.7402 - val_loss: 80.0016 - val_mae: 6.7239 
Epoch 9/20 356/356 - 35s - loss: 33.5237 - mae: 4.4271 - val_loss: 83.2579 - val_mae: 6.8529 
Epoch 10/20 356/356 - 35s - loss: 28.5170 - mae: 4.1411 - val_loss: 83.5056 - val_mae: 6.9629 
Epoch 11/20 356/356 - 35s - loss: 27.0142 - mae: 3.9700 - val_loss: 92.1290 - val_mae: 7.1866 
Epoch 12/20 356/356 - 35s - loss: 27.4564 - mae: 4.0428 - val_loss: 185.6307 - val_mae: 11.4591 
Epoch 13/20 356/356 - 35s - loss: 23.7961 - mae: 3.7407 - val_loss: 92.3429 - val_mae: 7.2467 
Epoch 14/20 356/356 - 35s - loss: 24.6167 - mae: 3.8116 - val_loss: 92.4542 - val_mae: 7.1401 
Epoch 15/20 356/356 - 35s - loss: 22.2604 - mae: 3.6746 - val_loss: 82.5822 - val_mae: 6.7841 
Epoch 16/20 356/356 - 35s - loss: 20.1899 - mae: 3.4430 - val_loss: 86.3830 - val_mae: 6.8304 
Epoch 17/20 356/356 - 35s - loss: 17.3425 - mae: 3.2205 - val_loss: 78.4369 - val_mae: 6.6419 
Epoch 18/20 356/356 - 35s - loss: 16.5249 - mae: 3.1295 - val_loss: 81.7731 - val_mae: 6.7226 
Epoch 19/20 356/356 - 35s - loss: 16.6140 - mae: 3.1421 - val_loss: 80.9727 - val_mae: 6.9908 
Epoch 20/20 356/356 - 35s - loss: 17.0187 - mae: 3.1785 - val_loss: 93.4115 - val_mae: 7.6512
'''

# conclusion
'''
After 20 epochs, training is significantly outperforming validation, suggesting that the model is overfitted. 
Furthermore, I'm not sure how happy the client will be with a 7-8 year average error. 
A better model might have been logistic comparing <21 with >=21 as these types of models tend to be more accurate and would still meet the client's needs. 
More rows of data might simultaneously decrease fitting and standard error.
Computer vision might be better for identifying and auto-tabulating items in a cart for faster checkout
'''