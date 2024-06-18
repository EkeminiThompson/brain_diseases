from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input
import joblib

app = Flask(__name__)

# Define custom mse function
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Load and preprocess data
def load_data():
    combined_data = pd.read_csv('combined_data.csv')
    imaging_data = np.load('imaging_data.npy')

    # Split features and labels
    X = combined_data.drop('disease_label', axis=1)
    y = combined_data['disease_label']
    
    # Preprocess the data (standardize, normalize, etc.)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, imaging_data, y

# Train and save RandomForest model
def train_rf_model(X_train, y_train):
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, 'rf_model.pkl')
    return rf_model

# Train and save Logistic Regression model for early diagnosis
def train_lr_model(X_train, y_train):
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, 'lr_model.pkl')
    return lr_model

# Train and save VAE model
def train_vae_model(X_train):
    input_dim = X_train.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(64, activation='relu')(input_layer)
    latent = Dense(2, activation='relu')(encoder)
    decoder = Dense(64, activation='relu')(latent)
    output_layer = Dense(input_dim, activation='sigmoid')(decoder)
    
    vae_model = Model(input_layer, output_layer)
    vae_model.compile(optimizer='adam', loss=mse)
    vae_model.fit(X_train, X_train, epochs=50, batch_size=32)
    vae_model.save('vae_model.h5')
    return vae_model

# Validate the model
def validate_model(model, X_val, y_val):
    predictions = model.predict(X_val)
    return classification_report(y_val, predictions, output_dict=True)

@app.route('/')
def index():
    X, imaging_data, y = load_data()
    return render_template('index.html', data_shape=X.shape)

# Train and save CNN model
def train_cnn_model(imaging_data, y_train):
    cnn_model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 1)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnn_model.fit(imaging_data, y_train, epochs=10, batch_size=32)
    cnn_model.save('cnn_model.h5')
    return cnn_model

@app.route('/train', methods=['GET', 'POST'])
def train_models():
    if request.method == 'POST':
        X, imaging_data, y = load_data()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(imaging_data, y, test_size=0.2, random_state=42)
        
        rf_model = train_rf_model(X_train, y_train)
        cnn_model = train_cnn_model(X_train_cnn, y_train_cnn)
        lr_model = train_lr_model(X_train, y_train)
        vae_model = train_vae_model(X_train)
        
        rf_report = validate_model(rf_model, X_test, y_test)
        lr_report = validate_model(lr_model, X_test, y_test)
        
        reports = {
            'RandomForest Report': rf_report,
            'LogisticRegression Report': lr_report
        }
        
        return render_template('train.html', reports=reports)
    else:
        return render_template('train.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    sample_genomic_data = []
    sample_clinical_data = []
    sample_biomarker_data = []
    sample_behavioral_data = []
    sample_environmental_data = []

    if request.method == 'POST':
        try:
            # Load the saved models
            rf_model = joblib.load('rf_model.pkl')
            lr_model = joblib.load('lr_model.pkl')
            vae_model = load_model('vae_model.h5', custom_objects={'mse': mse})

            # Check if form data is empty or contains only zeros
            if not request.form or all(val == '' or val == '0' for val in request.form.values()):
                sample_genomic_data = np.random.rand(1, 500).flatten().tolist()
                sample_clinical_data = np.random.rand(1, 3).flatten().tolist()
                sample_biomarker_data = np.random.rand(1, 100).flatten().tolist()
                sample_behavioral_data = np.random.rand(1, 50).flatten().tolist()
                sample_environmental_data = np.random.rand(1, 20).flatten().tolist()
                
                genomic_data = sample_genomic_data
                clinical_data = sample_clinical_data
                imaging_data = [0] * 4096  # Placeholder for imaging data
                biomarker_data = sample_biomarker_data
                behavioral_data = sample_behavioral_data
                environmental_data = sample_environmental_data
            else:
                # Get the form data
                genomic_data = [float(request.form.get(f'genomic_{i}')) for i in range(500)]
                clinical_data = [float(request.form.get(f'clinical_{i}')) for i in range(3)]
                imaging_data = [float(request.form.get(f'imaging_{i}')) for i in range(4096)]
                biomarker_data = [float(request.form.get(f'biomarker_{i}')) for i in range(100)]
                behavioral_data = [float(request.form.get(f'behavioral_{i}')) for i in range(50)]
                environmental_data = [float(request.form.get(f'environmental_{i}')) for i in range(20)]

            # Preprocess the data
            scaler = StandardScaler()
            genomic_data_scaled = scaler.fit_transform([genomic_data])
            clinical_data_scaled = scaler.fit_transform([clinical_data])
            imaging_data_reshaped = np.array(imaging_data).reshape(1, 64, 64, 1)
            biomarker_data_scaled = scaler.fit_transform([biomarker_data])
            behavioral_data_scaled = scaler.fit_transform([behavioral_data])
            environmental_data_scaled = scaler.fit_transform([environmental_data])

            # Make predictions
            X = np.concatenate((genomic_data_scaled, clinical_data_scaled, biomarker_data_scaled, behavioral_data_scaled, environmental_data_scaled), axis=1)
            rf_prediction = rf_model.predict(X)
            lr_prediction = lr_model.predict(X)
            vae_prediction = vae_model.predict(X)

            # Format the predictions
            rf_prediction_str = 'Yes' if rf_prediction[0] == 1 else 'No'
            lr_prediction_str = 'Yes' if lr_prediction[0] == 1 else 'No'
            vae_prediction_str = 'Yes' if vae_prediction[0][0] == 1 else 'No'

            predictions = {
                'RandomForest': rf_prediction_str,
                'LogisticRegression': lr_prediction_str,
                'VAE': vae_prediction_str
            }

            return render_template('prediction_result.html', predictions=predictions)
        except Exception as e:
            return f"An error occurred: {e}"
    else:
        # Return the form template with sample data for easier testing
        return render_template('predict.html',
                               sample_genomic_data=sample_genomic_data,
                               sample_clinical_data=sample_clinical_data,
                               sample_biomarker_data=sample_biomarker_data,
                               sample_behavioral_data=sample_behavioral_data,
                               sample_environmental_data=sample_environmental_data)
@app.route('/fill_data', methods=['GET'])
def fill_data():
    # Pre-fill the form data
    pre_filled_data = {
        'genomic_0': 0.5,
        'genomic_1': 0.3,
        'clinical_0': 25.0,
        'clinical_1': 120.0,
        'clinical_2': 180.0,
        'biomarker_0': 0.8,
        'biomarker_1': 0.7,
        'behavioral_0': 0.6,
        'behavioral_1': 0.4,
        'environmental_0': 0.2,
        'environmental_1': 0.1
    }

    return render_template('fill_data.html', data=pre_filled_data)


if __name__ == '__main__':
    app.run(debug=True)
