import pandas as pd
import joblib

MODEL_PATH = "car_price_model.joblib"  # correct model file

def predict_price(brand, cartype, fueltype, enginesize,status):
    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Create input DataFrame EXACTLY as training data
    new_car = pd.DataFrame({
        "brand": [brand],
        "car_type": [cartype],
        "fuel_type": [fueltype],
        "engine": [enginesize],
        "status" : [status]
    })

    predicted_price = model.predict(new_car)[0]
    return round(predicted_price, 2)
