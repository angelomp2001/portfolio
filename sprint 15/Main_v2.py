'''facial recognition model to predict a person's age'''

# libraries
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# Modules (files)
# from Interfaces.DataLoader import DataLoader
# from EDA.DataStructure import DataStructureAnalyzer
# from Interfaces.Outputter import OutPut

### Project/Interfaces/DataLoader/
import pandas as pd
import os

### Use_cases/Data_Modeller.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam

class DataLoader:
    @staticmethod # allows you to call a method without creating an instance of the class
    def from_csv(path: str) -> pd.DataFrame:
        return pd.read_csv(path)
    
    @staticmethod
    def from_path(path: str) -> list:
        file_list = os.listdir(path)

        # Filter for image files 
        image_files = [file for file in file_list if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

        # image file paths
        return [os.path.join(path, image) for image in image_files]
                  
### Project/EDA/DataStructure/
# class for exploring structure of data

class DataStructureAnalyzer:
    def __init__(self, data):
        self.data = data
        self.numerical = data.select_dtypes(include=['int64', 'float64']).columns.tolist()  # Use numeric types
        self.categorical = data.select_dtypes(include=['object']).columns.tolist()  # Object types
        self.time = data.select_dtypes(include=['datetime64[ns]']).columns.tolist()  # Datetime types

    def overview(self):
        return {
            'n_rows': self.data.shape[0],
            'n_columns': self.data.shape[1],
            'columns': self.data.columns.tolist(),
            'numerical_columns': self.numerical,
            'categorical_columns': self.categorical,
            'time_columns': self.time,
        }
    
    def dtypes(self):
        return self.data.dtypes.to_dict()
    
    def missing(self):
        return self.data.isnull().sum().to_dict()
    
    def nunique(self):
        return self.data.nunique().to_dict()
    
### Interfaces/Outputter/
import pandas as pd
import matplotlib.pyplot as plt

class OutPut:
    def to_console(self, data:None):
        self.data = data
        """Print data to console."""
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"{key}: {value}")
                 
        elif isinstance(data, pd.DataFrame | pd.Series):
            print(data.head(5))
            
        elif isinstance(data, list):
            self.show_image(data)

        else:
            print(f'dict, pd.Dataframe/Series required')

        return self
            
    def show_image(self, data: list):
        # view sample images
        show = 10
        
        fig, axs = plt.subplots(2, 5, figsize = (10,10))
        axs = axs.flatten()

        for n, img in enumerate(data[:show]):
            sample_image = Image.open(img)
            axs[n].imshow(sample_image)
            axs[n].axis('off')
            plt.tight_layout()
            
        plt.show()
            

    def view(self):
        """View all columns appropriately based on dtype."""
        for col in self.data.columns:
            dtype = self.data[col].dtype

            print(f"\n--- Viewing column: {col} ({dtype}) ---")

            if pd.api.types.is_numeric_dtype(dtype):
                OutPut._plot_numeric(self.data, col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                OutPut._plot_time(self.data, col)
            elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
                OutPut._plot_categorical(self.data, col)
            else:
                print(f"Unsupported dtype ({dtype}), showing raw values:")
                print(self.data[col].head())

    # --- Private helper methods ---
    @staticmethod
    def _plot_numeric(df, col):
        df[col].plot(kind='hist', bins=30, title=f"{col} (Numeric Distribution)")
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

    @staticmethod
    def _plot_categorical(df, col):
        value_counts = df[col].value_counts().head(20)
        value_counts.plot(kind='bar', title=f"{col} (Top 20 Categories)")
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.show()

    @staticmethod
    def _plot_time(df, col):
        df[col].value_counts().sort_index().plot(kind='line', title=f"{col} (Over Time)")
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.show()

class DataCleaner:
    def __init__(self, data):
        self.data = data

    def drop(self, col):
        self.data.drop(col, inplace = True, axis = 1)

# Use_cases/Data_Modelling/
class DataModeller:
    def __init__(self, target_size = (244, 244), mode = 'regression'):
        self.target_size = target_size
        self.input_shape = (target_size[0], target_size[1], 3)
        self.train_data = None
        self.test_data = None
        self.model = None
        self.mode = mode
        self.class_mode = None
        
        if self.mode == 'regression':
            class_mode = 'raw'
        else:
            class_mode = 'categorical' if self.mode == 'multiclass' else 'binary'

        self.class_mode = class_mode


    def load_data(
            self, 
            labels_df, 
            image_path, 
            x_col,
            y_col,
            target_size = (244, 244), 
            validation_split = 0.2, 
            batch_size = 16,  
            seed = 12345):
        """
        It loads the train part of dataset from path
        """
        self.target_size = target_size
        self.input_shape = (target_size[0], target_size[1], 3)

        

        train_gen = ImageDataGenerator(
            validation_split = validation_split,
            rescale = 1./255
            # horizontal_flip=True,
            # vertical_flip=True,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            # rotation_range=90
        )
        
        self.train_data = train_gen.flow_from_dataframe(
            dataframe = labels_df,
            directory = image_path,
            x_col = x_col,
            y_col = y_col,
            target_size = target_size,
            batch_size = batch_size,
            class_mode = self.class_mode, #'sparse',
            subset = 'training',
            seed = seed
        )
  

        # test data:
        test_gen = ImageDataGenerator(
            validation_split = validation_split,
            rescale=1./255
            # horizontal_flip=True,
            # vertical_flip=True,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            # rotation_range=90
        )
        
        self.test_data = test_gen.flow_from_dataframe(
            dataframe = labels_df,
            directory = image_path,
            x_col = x_col,
            y_col = y_col,
            target_size = target_size,
            batch_size = batch_size,
            class_mode = self.class_mode, #'sparse',
            subset = 'validation',
            seed = seed
        ) 
    
    # Define model (ResNet50)
    def create_model_ResNet50(self, n_classes = None):  
        backbone = ResNet50(weights='imagenet', 
                            input_shape = self.input_shape,
                            include_top=False)

        model = Sequential()
        model.add(backbone)
        model.add(GlobalAveragePooling2D())

        if self.mode == 'regression':
            model.add(Dense(1, activation='linear'))
            loss = 'mse'
            metrics = ['mae']
        elif self.mode == 'binary':
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:  # multiclass
            model.add(Dense(n_classes, activation='softmax'))
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']

        model.add(Dense(1, activation='linear'))

        optimizer = Adam(learning_rate=0.0005)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model = model
    
    # train model
    def train_model(self, batch_size=None, epochs=20,
                    steps_per_epoch=None, validation_steps=None):

        """
        Trains the model given the parameters
        """
        
        if steps_per_epoch is None:
            steps_per_epoch = len(self.train_data)
            
        if validation_steps is None:
            validation_steps = len(self.test_data)

        self.model.fit(
            self.train_data, 
            validation_data=self.test_data,
            batch_size=batch_size, epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            verbose=2)



# Main:
if __name__ == "__main__":
    # load data
    image_path = r'data/faces/'
    labels_path = r'data/faces/labels.csv'
    labels_df = DataLoader.from_csv(labels_path) # ['file_name','real_age']
    image_file_paths = DataLoader.from_path(image_path)

    # EDA
    eda = DataStructureAnalyzer(labels_df)
    output = OutPut()

    # save the state of the df at the start
    start = eda.overview()

    # print missing
    output.to_console(eda.missing())

    # print dtypes
    output.to_console(eda.dtypes())

    # print nunique 
    output.to_console(eda.nunique())

    # print cols as graphs
    output.to_console(labels_df).view()

    # view images
    output.to_console(image_file_paths)

    # load cleaner
    edit = DataCleaner(labels_df)

    # drop column
    # edit.drop(['file_name'])

    # model
    model = DataModeller()

    model.load_data(
        labels_df = labels_df,
        image_path = image_path,
        x_col = labels_df.columns[0],
        y_col = labels_df.columns[1],
        target_size = (244, 244),
        validation_split = 0.2,
        batch_size = 32,  
        seed = 12345)

    model.create_model_ResNet50(n_classes = None)

    model.train_model(
        batch_size=None, 
        epochs=20,
        steps_per_epoch=None, 
        validation_steps=None
    )