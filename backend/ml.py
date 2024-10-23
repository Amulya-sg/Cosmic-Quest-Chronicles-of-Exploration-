import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Load your dataset
df = pd.read_csv('your_exoplanet_data.csv')

# Display unique values to understand the data
print("Unique values in mass_wrt before processing:", df['mass_wrt'].unique())
print("Unique values in radius_wrt before processing:", df['radius_wrt'].unique())

# Ensure there are no NaN values in mass_wrt and radius_wrt
df.dropna(subset=['mass_wrt', 'radius_wrt'], inplace=True)

# Create mappings for mass and radius
mass_mappings = {'Jupiter': 1898, 'Earth': 1}  # mass in Earth masses
radius_mappings = {'Jupiter': 69911, 'Earth': 6371}  # radius in km

# Function to calculate mass and radius
def calculate_mass(row):
    if row['mass_wrt'] in mass_mappings:
        return row['mass_multiplier'] * mass_mappings[row['mass_wrt']]
    return np.nan

def calculate_radius(row):
    if row['radius_wrt'] in radius_mappings:
        return row['radius_multiplier'] * radius_mappings[row['radius_wrt']]
    return np.nan

# Apply the functions
df['mass'] = df.apply(calculate_mass, axis=1)
df['radius'] = df.apply(calculate_radius, axis=1)

# Drop rows where mass or radius is NaN
df.dropna(subset=['mass', 'radius'], inplace=True)

# Define features and target
X = df[['distance', 'stellar_magnitude', 'mass', 'radius', 'orbital_radius']]
y = df['planet_type']  # Assuming 'planet_type' is your target variable

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Check if we have data to split
if X.empty or y.size == 0:
    print("No data available for training and testing.")
else:
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and fit the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    print("Model training completed.")

    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    accuracy = (y_pred == y_test).mean()
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the trained model, scaler, and label encoder
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/random_forest_model.joblib')
    print("Random Forest model saved.")
    joblib.dump(scaler, 'models/scaler.joblib')
    print("Scaler saved.")
    joblib.dump(le, 'models/label_encoder.joblib')
    print("Label Encoder saved.")

    # Example prediction
    def predict_planet_type(mass_multiplier, mass_wrt, radius_multiplier, radius_wrt, distance, stellar_magnitude, orbital_radius):
        # Calculate mass and radius
        mass = mass_multiplier * mass_mappings.get(mass_wrt, np.nan)
        radius = radius_multiplier * radius_mappings.get(radius_wrt, np.nan)

        if np.isnan(mass) or np.isnan(radius):
            raise ValueError("Invalid mass_wrt or radius_wrt value.")

        # Prepare the feature array
        features = np.array([[distance, stellar_magnitude, mass, radius, orbital_radius]])

        # Scale the features
        features_scaled = scaler.transform(features)

        # Predict the planet type
        prediction = model.predict(features_scaled)
        planet_type = le.inverse_transform(prediction)[0]
        return planet_type

    # Example usage
    try:
        planet_type = predict_planet_type(1.5, 'Earth', 1.2, 'Jupiter', 100, 5, 1.5)
        print(f"The predicted planet type is: {planet_type}")
    except ValueError as e:
        print(e)