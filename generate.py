import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Set the paths for the imaging data
yes_path = r'C:\Users\EkeminiThompson\Downloads\brain_disease\yes'
no_path = r'C:\Users\EkeminiThompson\Downloads\brain_disease\no'

# Function to load and preprocess images
def load_images(image_folder, label, img_size=(64, 64)):
    images = []
    labels = []
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg'):
            img_path = os.path.join(image_folder, filename)
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            image = image.resize(img_size)
            image_array = np.array(image)
            images.append(image_array)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load images and labels
yes_images, yes_labels = load_images(yes_path, 1)
no_images, no_labels = load_images(no_path, 0)

# Combine yes and no images
imaging_data = np.concatenate((yes_images, no_images), axis=0)
labels = np.concatenate((yes_labels, no_labels), axis=0)

# Normalize the imaging data
imaging_data = imaging_data / 255.0  # Scale pixel values to [0, 1]

# Reshape for CNN model
imaging_data = imaging_data.reshape(imaging_data.shape[0], 64, 64, 1)

# Save imaging data to a numpy file
np.save('imaging_data.npy', imaging_data)

# Print a message indicating that the imaging data has been generated
print("Imaging data saved as 'imaging_data.npy'")

# Generate synthetic genomic data
num_samples = imaging_data.shape[0]
num_genomic_features = 500
genomic_feature_names = [f'genomic_feature_{i}' for i in range(1, num_genomic_features + 1)]
genomic_data = np.random.rand(num_samples, num_genomic_features)

# Generate synthetic clinical data
num_clinical_features = 3
clinical_feature_names = ['age', 'blood_pressure', 'cholesterol']
clinical_data = np.random.rand(num_samples, num_clinical_features)

# Generate synthetic biomarker data
num_biomarker_features = 100
biomarker_feature_names = [f'biomarker_feature_{i}' for i in range(1, num_biomarker_features + 1)]
biomarker_data = np.random.rand(num_samples, num_biomarker_features)

# Generate synthetic behavioral data
num_behavioral_features = 50
behavioral_feature_names = [f'behavioral_feature_{i}' for i in range(1, num_behavioral_features + 1)]
behavioral_data = np.random.rand(num_samples, num_behavioral_features)

# Generate synthetic environmental data
num_environmental_features = 20
environmental_feature_names = [f'environmental_feature_{i}' for i in range(1, num_environmental_features + 1)]
environmental_data = np.random.rand(num_samples, num_environmental_features)

# Combine all synthetic data into a DataFrame
genomic_df = pd.DataFrame(genomic_data, columns=genomic_feature_names)
clinical_df = pd.DataFrame(clinical_data, columns=clinical_feature_names)
biomarker_df = pd.DataFrame(biomarker_data, columns=biomarker_feature_names)
behavioral_df = pd.DataFrame(behavioral_data, columns=behavioral_feature_names)
environmental_df = pd.DataFrame(environmental_data, columns=environmental_feature_names)
labels_df = pd.DataFrame(labels, columns=['disease_label'])

# Combine all data into a single DataFrame
combined_data = pd.concat([genomic_df, clinical_df, biomarker_df, behavioral_df, environmental_df, labels_df], axis=1)

# Save combined data to a CSV file
combined_data.to_csv('combined_data.csv', index=False)

# Print shapes of the generated data
print(f"Genomic data shape: {genomic_data.shape}")
print(f"Clinical data shape: {clinical_data.shape}")
print(f"Imaging data shape: {imaging_data.shape}")
print(f"Biomarker data shape: {biomarker_data.shape}")
print(f"Behavioral data shape: {behavioral_data.shape}")
print(f"Environmental data shape: {environmental_data.shape}")
print(f"Labels shape: {labels.shape}")
