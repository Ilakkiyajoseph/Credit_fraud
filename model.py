import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import category_encoders as ce
import numpy as np
import joblib

def preprocess(data):
    """
    Applies Frequency Encoding to categorical variables and Binary Encoding to binary variables.
    Also transforms date of birth into age.
    
    Parameters:
    data (pd.DataFrame): The input DataFrame with categorical and date of birth columns.
    
    Returns:
    pd.DataFrame: The transformed and encoded DataFrame for training.
    pd.DataFrame: The transformed and encoded DataFrame for testing.
    ce.TargetEncoder: The fitted encoder.
    StandardScaler: The fitted scaler.
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

    # Transform 'dob' into 'age' if 'dob' column exists
    if 'dob' in data.columns:
        data['dob'] = pd.to_datetime(data['dob'], errors='coerce')
        current_year = pd.to_datetime('today').year
        data['age'] = current_year - data['dob'].dt.year

    # Drop the original 'dob' column as it has been converted to 'age'
    if 'dob' in data.columns:
        data = data.drop('dob', axis=1)

    # Ensure 'is_fraud' column is binary and of integer type
    if 'is_fraud' in data.columns:
        data['is_fraud'] = data['is_fraud'].astype(int)

    # Separate out numerical columns for scaling
    numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # Separate out categorical columns for target encoding
    categorical_columns = data.select_dtypes(include=['object']).columns
    encoder = ce.TargetEncoder(cols=categorical_columns)

    # Fit and transform on the entire data
    data_encoded = encoder.fit_transform(data, data['is_fraud'])

    # Check and handle missing or infinite values
    if data_encoded.isna().sum().sum() > 0:
        data_encoded = data_encoded.fillna(0)
    if np.isinf(data_encoded).sum().sum() > 0:
        data_encoded = data_encoded.replace([np.inf, -np.inf], 0)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(data_encoded.drop('is_fraud', axis=1), data_encoded['is_fraud'], test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, encoder, scaler

def train_model(X_train, X_test, y_train, y_test):
    """
    Train and evaluate a Random Forest model.

    Parameters:
    X_train (pd.DataFrame): Training features.
    X_test (pd.DataFrame): Testing features.
    y_train (pd.Series): Training target.
    y_test (pd.Series): Testing target.

    Returns:
    RandomForestClassifier: Trained model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=123)
    model.fit(X_train, y_train)
    return model

def save_model(model, encoder, scaler, filename_model, filename_encoder, filename_scaler):
    """
    Save the trained model, encoder, and scaler to files.

    Parameters:
    model: Trained model object.
    encoder: Fitted encoder object.
    scaler: Fitted scaler object.
    filename_model (str): File path to save the model.
    filename_encoder (str): File path to save the encoder.
    filename_scaler (str): File path to save the scaler.
    """
    joblib.dump(model, filename_model)
    joblib.dump(encoder, filename_encoder)
    joblib.dump(scaler, filename_scaler)

def main():
    # Assuming data_cleaned is a DataFrame containing the cleaned data with an 'is_fraud' column
    data_cleaned = pd.read_csv('cleaned_data.csv')  # Replace with your data loading logic

    # Preprocess data
    X_train, X_test, y_train, y_test, encoder, scaler = preprocess(data_cleaned)

    # Train the model
    model = train_model(X_train, X_test, y_train, y_test)

    # Save the model, encoder, and scaler
    save_model(model, encoder, scaler, 'random_forest_model.pkl', 'encoder.pkl', 'scaler.pkl')

    print("Model training and saving completed.")

if __name__ == "__main__":
    main()
