''' business logic for the API endpoints '''
import joblib
import numpy as np
import json
from typing import Dict, Any
import os

# Global variable to store the loaded model
model = None
metadata = None

def load_model_and_metadata():
    """
    Load the trained model and metadata from disk.
    This function should:
    1. Load model.pkl using joblib.load()
    2. Load model_metadata.json using json.load()
    3. Store them in the global variables
    4. Return True if successful, False if there's an error
    """
    global model, metadata

    try:
        model = joblib.load('model.pkl')
        # YOUR TASK: Load the metadata
        # Open 'model_metadata.json' and load it with json.load()
        # Store it in the global 'metadata' variable
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)

        print("Model and metadata loaded successfully!")
        return True

    except Exception as e:
        print(f"Error loading model or metadata: {e}")
        return False

def make_prediction(house_features: Dict[str, Any]) -> float:
    """
    Make a price prediction for a single house.

    Args:
        house_features: Dictionary containing all 13 features

    Returns:
        Predicted price as a float

    This function should:
    1. Extract the features in the correct order (same order as training)
    2. Convert to numpy array with shape (1, 13)
    3. Use model.predict() to get prediction
    4. Return the prediction as a float
    """
    global model

    if model is None:
        raise ValueError("Model not loaded")

    # YOUR TASK: Extract features in the correct order
    # The order MUST match the order in metadata['features']
    # Hint: Use a list comprehension to get features in order
    feature_values = [house_features[feature] for feature in metadata['features']]

    # YOUR TASK: Convert to numpy array with correct shape
    # The model expects shape (1, 13) - one row, 13 columns
    X = np.array(feature_values).reshape(1, -1)

    # YOUR TASK: Make prediction and extract single value
    # model.predict() returns an array, we need the single value
    prediction = model.predict(X)[0]

    # Round to 2 decimal places for currency
    return round(float(prediction), 2)

def get_model_info() -> Dict[str, Any]:
    """
    Get information about the loaded model.

    Returns:
        Dictionary containing model metadata

    This function should simply return the loaded metadata dictionary.
    If metadata is not loaded, it should raise an error.
    """
    global metadata

    # YOUR TASK: Check if metadata is loaded and return it
    # If metadata is None, raise ValueError("Model metadata not loaded")
    if metadata is None:
        raise ValueError("Model metadata not loaded")
    return metadata


def check_health() -> Dict[str, Any]:
    """
    Check the health status of the service.

    Returns:
        Dictionary with health status information

    This function should:
    1. Check if model is loaded (model is not None)
    2. Check if metadata is loaded (metadata is not None)
    3. Return a dictionary with status information
    """
    global model, metadata

    # YOUR TASK: Create health status dictionary
    # Include:
    # - "status": "healthy" if both model and metadata are loaded, "unhealthy" otherwise
    # - "model_loaded": True/False based on whether model is not None
    # - "message": Descriptive message about the status

    health_status = {
        "status": "healthy" if model is not None and metadata is not None else "unhealthy",
        "model_loaded": model is not None,
        "message": "The status of the model"  # Update with appropriate message
    }

    return health_status
