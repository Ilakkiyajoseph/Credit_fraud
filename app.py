from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Load the encoder and scaler
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

def preprocess(data, encoder, scaler):
    """
    Applies Frequency Encoding to categorical variables and Binary Encoding to binary variables.
    Also transforms date of birth into age.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame with categorical and date of birth columns.
    encoder (ce.TargetEncoder): The fitted target encoder.
    scaler (StandardScaler): The fitted scaler.
    
    Returns:
    pd.DataFrame: The transformed and encoded DataFrame.
    """

    # Frequency Encoding for 'category', 'city', and 'state'
    if 'category' in data.columns:
        data['category'] = data['category'].map(data['category'].value_counts(normalize=True))
    if 'city' in data.columns:
        data['city'] = data['city'].map(data['city'].value_counts(normalize=True))
    if 'state' in data.columns:
        data['state'] = data['state'].map(data['state'].value_counts(normalize=True))

    # Binary Encoding for 'gender'
    if 'gender' in data.columns:
        data['gender'] = data['gender'].map({'F': 0, 'M': 1})

    # Separate out numerical columns for scaling
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numerical_columns] = scaler.transform(data[numerical_columns])

    return data

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        form_data = request.json
        print("Raw form data:", form_data)  # Debugging output

        data = {
            'cc_num': [int(form_data.get('cc_num'))],
            'category': [form_data.get('category')],
            'amt': [float(form_data.get('amt', 0))],
            'gender': [form_data.get('gender')],
            'city': [form_data.get('city')],
            'state': [form_data.get('state')],
            'zip': [int(form_data.get('zip'))],
            'lat': [float(form_data.get('lat', 0))],
            'long': [float(form_data.get('long', 0))],
            'city_pop': [int(form_data.get('city_pop', 0))],
            'merch_lat': [float(form_data.get('merch_lat', 0))],
            'merch_long': [float(form_data.get('merch_long', 0))],
            'trans_year': [int(form_data.get('trans_year', 0))],
            'trans_month': [int(form_data.get('trans_month', 0))],
            'trans_date': [int(form_data.get('trans_date', 0))],
            'trans_hour': [int(form_data.get('trans_hour', 0))],
            'trans_min': [int(form_data.get('trans_min', 0))],
            'trans_seconds': [int(form_data.get('trans_seconds', 0))],
            'age': [int(form_data.get('age', 0))]
        }
        df = pd.DataFrame(data)
        print("DataFrame before preprocessing:", df)  # Debugging output
        
        # Preprocess input data using the loaded encoder and scaler
        processed_data = preprocess(df, encoder, scaler)
        print("Processed data:", processed_data)  # Debugging output

        # Make prediction
        prediction = model.predict(processed_data)
        
        # Return prediction as JSON response
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        print("Error:", e)  # Debugging output
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
