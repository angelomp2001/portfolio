from pydantic import BaseModel, Field
from typing import Optional

class HousePredictionRequest(BaseModel):
    """Schema for house prediction request - validates all 13 features"""

    # Property listing features
    total_images: int = Field(..., ge=0, le=50, description="Number of property images (0-50)")

    # Basic property features
    beds: int = Field(..., ge=0, le=10, description="Number of bedrooms (0-10)")
    baths: float = Field(..., ge=0, le=10, description="Number of bathrooms (0-10, can be decimal like 2.5)")
    area: float = Field(..., gt=0, le=10000, description="Property area in square feet (must be positive, max 10000)")

    # Location features
    latitude: float = Field(..., ge=-90, le=90, description="Property latitude (-90 to 90)")
    longitude: float = Field(..., ge=-180, le=180, description="Property longitude (-180 to 180)")

    # Binary features (0 or 1)
    garden: int = Field(..., ge=0, le=1, description="Has garden: 0=No, 1=Yes")
    garage: int = Field(..., ge=0, le=1, description="Has garage: 0=No, 1=Yes")
    new_construction: int = Field(..., ge=0, le=1, description="Is new construction: 0=No, 1=Yes")
    pool: int = Field(..., ge=0, le=1, description="Has pool: 0=No, 1=Yes")
    terrace: int = Field(..., ge=0, le=1, description="Has terrace: 0=No, 1=Yes")
    air_conditioning: int = Field(..., ge=0, le=1, description="Has AC: 0=No, 1=Yes")
    parking: int = Field(..., ge=0, le=1, description="Has parking: 0=No, 1=Yes")

    class Config:
        schema_extra = {
            "example": {
                "total_images": 10,
                "beds": 3,
                "baths": 2.5,
                "area": 1800.0,
                "latitude": 40.7128,
                "longitude": -74.0060,
                "garden": 1,
                "garage": 1,
                "new_construction": 0,
                "pool": 0,
                "terrace": 1,
                "air_conditioning": 1,
                "parking": 1
            }
        }

# YOUR TASK: Add the PredictionResponse schema
# This schema defines what your API returns after making a prediction
# It should have:
# - predicted_price: float (the predicted house price)
# - currency: str with default value "USD"
# - model_version: str (the version of the model used)

class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    # Add the three fields here
    # Remember to use Field() to add descriptions!
    pass  # Remove this line when you add the fields

# YOUR TASK: Add the ModelInfoResponse schema
# This schema defines what the /model/info endpoint returns
# It should have:
# - model_type: str
# - version: str
# - features: list of strings
# - training_date: str
# - rmse: float
# - description: str

class ModelInfoResponse(BaseModel):
    """Schema for model information response"""
    # Add all six fields here
    # Remember to use Field() to add descriptions!
    pass  # Remove this line when you add the fields

# YOUR TASK: Add the HealthCheckResponse schema
# This schema defines what the /health endpoint returns
# It should have:
# - status: str (should be "healthy" or "unhealthy")
# - model_loaded: bool (True if model is loaded, False otherwise)
# - message: str (a descriptive message)

class HealthCheckResponse(BaseModel):
    """Schema for health check response"""
    # Add the three fields here
    pass  # Remove this line when you add the fields
