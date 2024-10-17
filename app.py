from flask import Flask, render_template, request
import requests
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

app = Flask(__name__)

# API configuration
rapid_api_host = "rto-vehicle-information-verification-india.p.rapidapi.com"
rapid_api_key = "aaf5746cb7msh1583ee475f18d7ep10a932jsne18292a68c87"

# Training and prediction function
def train_and_predict(car_name, brand, model, fuel_type, km_driven, seller_type, max_power, seats, engine, vehicle_age, ex_showroom_price, df):
    # Log-transform for skewed features
    df['log_km_driven'] = np.log1p(df['km_driven'])
    df['log_engine'] = np.log1p(df['engine'])

    # Interaction features
    df['age_km_interaction'] = df['vehicle_age'] * df['km_driven']

    # Feature selection
    X = df[['car_name', 'brand', 'model', 'log_km_driven', 'fuel_type', 'seller_type', 'max_power', 'seats', 'log_engine', 'vehicle_age', 'age_km_interaction']]
    y = df['selling_price']

    # Column transformer for preprocessing
    column_trans = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), ['car_name', 'brand', 'model', 'fuel_type', 'seller_type']),
        (StandardScaler(), ['log_km_driven', 'max_power', 'seats', 'log_engine', 'vehicle_age', 'age_km_interaction']),
        remainder='passthrough'
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=293)

    # XGBRegressor and hyperparameter tuning
    xgb = XGBRegressor()
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }

    grid_search = GridSearchCV(xgb, param_grid, cv=3, n_jobs=-1, verbose=1)
    
    # Pipeline with preprocessor and model
    pipe = make_pipeline(column_trans, grid_search)
    pipe.fit(X_train, y_train)

    # Input data for prediction
    input_data = pd.DataFrame({
        'car_name': [car_name],
        'brand': [brand],
        'model': [model],
        'log_km_driven': [np.log1p(km_driven)],
        'fuel_type': [fuel_type],
        'seller_type': [seller_type],
        'max_power': [max_power],
        'seats': [seats],
        'log_engine': [np.log1p(engine)],
        'vehicle_age': [vehicle_age],
        'age_km_interaction': [vehicle_age * km_driven]
    })

    # Predict the price
    predicted_price = pipe.predict(input_data)[0]

    # Cap the predicted price at the ex-showroom price
    return min(predicted_price, ex_showroom_price)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/car-login')
def car_login():
    return render_template('car-login.html')

@app.route('/submit-car-number', methods=['POST'])
def submit_car_number():
    car_number = request.form['car_number']
    km_driven = request.form['km_driven']  # Get km_driven from the form

    # Call RapidAPI to fetch car details
    url = "https://rto-vehicle-information-verification-india.p.rapidapi.com/api/v1/rc/vehicleinfo"

    payload = {
        "reg_no": car_number,
        "consent": "Y",
        "consent_text": "I hereby declare my consent agreement for fetching my information via AITAN Labs API"
    }
    headers = {
        "x-rapidapi-key": rapid_api_key,
        "x-rapidapi-host": rapid_api_host,
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        car_data1 = response.json()
        car_data = car_data1["result"]

        # Log the fetched car data to debug
        print(f"Full API response: {car_data}")

        # Load your dataset for prediction (assuming it's a CSV or a DataFrame)
        df = pd.read_csv('cardekho_dataset.csv')  # Replace with actual dataset

        # Fetch the ex-showroom price using the correct key 'sale_amount'
        ex_showroom_price = car_data.get('sale_amount', 0)  # Use 'sale_amount' for ex-showroom price
        print(f"Ex-showroom price: {ex_showroom_price}")

        # Validate ex-showroom price
        if ex_showroom_price == 0:
            return f"Invalid ex-showroom price for car number {car_number}. Please check the details."

        # Proceed with prediction if valid price exists
        predicted_price = train_and_predict(
            car_name=car_data['vehicle_manufacturer_name'] + " " + car_data['model'], 
            brand=car_data['vehicle_manufacturer_name'], 
            model=car_data['model'], 
            km_driven=int(km_driven),  # Use the km_driven value from the form
            fuel_type=car_data['fuel_descr'],  # "fuel_descr" mapped to fuel_type
            seller_type=car_data['owner_code_descr'],  # "owner_code_descr" mapped to seller_type
            max_power=car_data['vehicle_hp'],  # "vehicle_hp" mapped to max_power
            seats=car_data['vehicle_seat_capacity'],  # Assuming "vehicle_seat_capacity" corresponds to seats
            engine=car_data['cubic_cap'],  # "cubic_cap" mapped to engine
            vehicle_age=2024 - int(car_data['purchase_date'].split('-')[0]),  # Calculating vehicle age from "purchase_date"
            ex_showroom_price=ex_showroom_price,  # Use 'sale_amount' as ex-showroom price
            df=df
        )
        
        # Render the car details dashboard with fetched data and predicted price
        return render_template('car-details.html', car_data=car_data, predicted_price=predicted_price)
    else:
        return f"Failed to fetch details for car number {car_number}. Please try again.<br>Details: {response.text}"

if __name__ == '__main__':
    app.run(debug=True)
