# KEEP

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam
import pandas as pd
from tensorflow.keras.layers import Dropout

def train_image_generator_binary(
        train_images_directory,
        target_size,
        params,
        batch_size=16,
        validation_split=0.2,
        seed=12345):
    """
    Creates generators for BINARY classification (0/1).
    """
    train_datagen = ImageDataGenerator(
        validation_split=validation_split,
        rescale = params['rescale'],
        rotation_range = params['rotation_range'],
        width_shift_range = params['width_shift_range'],
        height_shift_range = params['height_shift_range'],
        shear_range = params['shear_range'],
        zoom_range = params['zoom_range'],
        horizontal_flip = params['horizontal_flip'],
        vertical_flip = params['vertical_flip'],
        fill_mode = params['fill_mode'],
        featurewise_center = params['featurewise_center'],
        samplewise_center = params['samplewise_center'],
        featurewise_std_normalization = params['featurewise_std_normalization'],
        samplewise_std_normalization = params['samplewise_std_normalization'],
        brightness_range = params['brightness_range'],
        channel_shift_range = params['channel_shift_range'],
        zca_whitening = params['zca_whitening'],
        zca_epsilon = params['zca_epsilon']
    )

    val_datagen = ImageDataGenerator(
        validation_split=validation_split,
        rescale = params['rescale'],
        rotation_range = params['rotation_range'],
        width_shift_range = params['width_shift_range'],
        height_shift_range = params['height_shift_range'],
        shear_range = params['shear_range'],
        zoom_range = params['zoom_range'],
        horizontal_flip = params['horizontal_flip'],
        vertical_flip = params['vertical_flip'],
        fill_mode = params['fill_mode'],
        featurewise_center = params['featurewise_center'],
        samplewise_center = params['samplewise_center'],
        featurewise_std_normalization = params['featurewise_std_normalization'],
        samplewise_std_normalization = params['samplewise_std_normalization'],
        brightness_range = params['brightness_range'],
        channel_shift_range = params['channel_shift_range'],
        zca_whitening = params['zca_whitening'],
        zca_epsilon = params['zca_epsilon']
    )

    train_flow = train_datagen.flow_from_directory(
        directory = train_images_directory,
        target_size=target_size,
        color_mode="rgb",       # or "grayscale"
        batch_size=batch_size,
        class_mode='binary',     
        subset='training',
        seed=seed
    )

    val_flow = val_datagen.flow_from_directory(
        directory = train_images_directory,
        target_size=target_size,
        color_mode="rgb",       
        batch_size=batch_size,
        class_mode='binary',  
        subset='validation',
        seed=seed
    )

    return train_flow, val_flow

def train_image_generator__multiclass(
        train_images_directory,
        target_size,
        validation_split,
        batch_size=16,
        seed=12345):
    """
    Creates generators for MULTI-CLASS classification.
    """
    train_datagen = ImageDataGenerator(
        validation_split=validation_split,
        rescale=1./255
    )
    val_datagen = ImageDataGenerator(
        validation_split=validation_split,
        rescale=1./255
    )

    train_flow = train_datagen.flow_from_directory(
        directory = train_images_directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='sparse',      
        subset='training',
        seed=seed
    )

    val_flow = val_datagen.flow_from_directory(
        directory = train_images_directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='sparse',      
        subset='validation',
        seed=seed
    )

    return train_flow, val_flow

def train_image_generator__linear(
        dataframe_csv,
        images_dir,
        target_size,
        validation_split,
        batch_size=16,
        seed=12345):
    """
    Creates generators for LINEAR regression (continuous target).
    Expects a CSV with columns: 'filename', 'real_age'
    """
    df = pd.read_csv(dataframe_csv)
    df['filepath'] = images_dir + '/' + df['filename']

    train_datagen = ImageDataGenerator(
        validation_split=validation_split,
        rescale=1./255
    )
    val_datagen = ImageDataGenerator(
        validation_split=validation_split,
        rescale=1./255
    )

    train_flow = train_datagen.flow_from_dataframe(
        dataframe=df,
        x_col='filepath',
        y_col='real_age',         
        target_size=target_size,
        batch_size=batch_size,
        class_mode='raw',         
        subset='training',
        seed=seed
    )

    val_flow = val_datagen.flow_from_dataframe(
        dataframe=df,
        x_col='filepath',
        y_col='real_age',
        target_size=target_size,
        batch_size=batch_size,
        class_mode='raw',
        subset='validation',
        seed=seed
    )

    return train_flow, val_flow

def build_model(input_shape, model_save_path):
  # Build AlexNet
  model = Sequential()
  model.add(Conv2D(6, (5, 5), padding='same', activation='relu', input_shape=input_shape)) # input_shape=(28, 28, 1)
  model.add(MaxPool2D(pool_size=(2, 2),strides=(1, 1))) # overlapping by 1
  model.add(Conv2D(16, (5, 5), padding='valid', activation='relu')) 
  model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1)))
  model.add(Flatten())
  model.add(Dense(120, activation='relu'))
  model.add(Dense(84, activation='relu'))
  # model.add(Dropout(0.5))
  # model.add(Dense(1, activation='sigmoid'))  # For binary classification
  model.add(Dense(10, activation='softmax'))
  # model.add(Dense(1, activation='linear'))  # for linear prediction

  # Compile
  model.compile(
      optimizer=Adam(learning_rate=0.001),
      # loss='binary_crossentropy', # for binary prediction
      loss='sparse_categorical_crossentropy',
      # loss='mean_squared_error' # for linear prediction
      metrics=['acc']
      # metrics=['mae']
  )

  checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True) #'best_model.keras'
  return model, checkpoint

def fit_model(
    model,                 # Keras model
    train_flow,            # Training data generator
    val_flow,              # Validation data generator
    model_save_path,       # model save location
    epochs: int = 10,      
    steps_per_epoch: int = None,
    validation_steps: int = None,
    callbacks=None
):
    """
    Fits a Keras model using image generators.
    """
    model.fit(
        train_flow,
        validation_data=val_flow,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2,
        epochs=epochs,
        callbacks=callbacks
    )
    model.save(model_save_path) #over_21_prediction_model.keras
    return model

def load_model(model_save_path):
    model = load_model(model_save_path)
    return model

def pred_image_generator_binary(
        params,
        test_image_path,
        target_shape):
    
    test_datagen = ImageDataGenerator(
        rescale = params['rescale'],
        rotation_range = params['rotation_range'],
        width_shift_range = params['width_shift_range'],
        height_shift_range = params['height_shift_range'],
        shear_range = params['shear_range'],
        zoom_range = params['zoom_range'],
        horizontal_flip = params['horizontal_flip'],
        vertical_flip = params['vertical_flip'],
        fill_mode = params['fill_mode'],
        featurewise_center = params['featurewise_center'],
        samplewise_center = params['samplewise_center'],
        featurewise_std_normalization = params['featurewise_std_normalization'],
        samplewise_std_normalization = params['samplewise_std_normalization'],
        brightness_range = params['brightness_range'],
        channel_shift_range = params['channel_shift_range'],
        zca_whitening = params['zca_whitening'],
        zca_epsilon = params['zca_epsilon']
    )

    # Load the image
    img = load_img(test_image_path, target_size=target_shape)  # Load and resize the image
    img_array = img_to_array(img)  # Convert to array

    # add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Use the datagen to apply transformations
    img_array = next(test_datagen.flow(img_array, batch_size=1)) 

    return img_array

def pred_target(model, img_array):
    pred = model.predict(img_array)

    # Make predictions
    pred = model.predict(img_array)

    predicted_class = (pred[0][0] > 0.5).astype("int")
    # predicted_class = np.argmax(pred, axis=1)[0] # for multiclass
    confidence = np.max(pred)

    print(f"Predicted class: {predicted_class}, confidence: {confidence:.2f}")

def main():
    params = {
        'rescale': 1./255,
        'rotation_range': 0, #40
        'width_shift_range': 0, #0.2,
        'height_shift_range': 0, #0.2,
        'shear_range': 0, #0.2,
        'zoom_range': 0, #0.2,
        'horizontal_flip': False, #True,
        'vertical_flip': False, #True,
        'fill_mode': 'nearest',
        'featurewise_center': False, #False,
        'samplewise_center': False, #False,
        'featurewise_std_normalization': False, #False,
        'samplewise_std_normalization': False, #False,
        'brightness_range': [0.0, 0.0], #[0.2, 1.0],
        'channel_shift_range': 0.0,
        'zca_whitening': False,
        'zca_epsilon': 1e-06,
    }

    train_image_path = 'data/faces/' 
    batch_size = 32
    target_size = (224, 224)
    input_shape = (target_size[0], target_size[1], 3)  # (Height, Width, Channels)
    validation_split = 0.20
    seed = 12345
    epochs=10
    test_image_path = 'data/linkedin headshot.jpeg'
    model_save_path = r'C:\Users\Angelo\Documents\vs code\models\computer vision\over_21_binary_prediction_model.keras'


    # Create data generators
    train_datagen_flow, val_datagen_flow = train_image_generator_binary(
        train_images_directory=train_image_path,
        target_size=target_size,
        params=params,
        seed=seed,
        batch_size=batch_size,
        validation_split=validation_split)

    # Create model
    model, checkpoint = build_model(input_shape, model_save_path)

    # Train the model
    fit_model(model, train_datagen_flow, val_datagen_flow, model_save_path=model_save_path ,epochs=epochs, callbacks=None) # callbacks=[checkpoint]

    # process test image
    image = pred_image_generator_binary(params,test_image_path,target_size)

    # predict label
    pred_target(model, image)

if __name__ == "__main__":
    main()

"The computer reboots when I try to run this."