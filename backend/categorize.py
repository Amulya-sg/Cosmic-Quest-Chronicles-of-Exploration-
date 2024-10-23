# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix

# # Load the dataset
# df = pd.read_csv('your_exoplanet_data.csv')

# # Replace problematic non-numeric values with NaN
# df.replace({'transist': np.nan}, inplace=True)

# # Define the mass and radius mapping
# mass_mapping = {
#     'Earth': 1,          # Mass of Earth
#     'Jupiter': 317.8,   # Mass of Jupiter
# }

# radius_mapping = {
#     'Earth': 1,          # Radius of Earth
#     'Jupiter': 11.2,    # Radius of Jupiter
# }

# # Check unique values in mass_wrt and radius_wrt before processing
# print("Unique values in mass_wrt before processing:", df['mass_wrt'].unique())
# print("Unique values in radius_wrt before processing:", df['radius_wrt'].unique())

# # Function to calculate actual mass
# def calculate_mass(row):
#     reference = row['mass_wrt']
#     multiplier = row['mass_multiplier']
#     return multiplier * mass_mapping.get(reference, np.nan)

# # Function to calculate actual radius
# def calculate_radius(row):
#     reference = row['radius_wrt']
#     multiplier = row['radius_multiplier']
#     return multiplier * radius_mapping.get(reference, np.nan)

# # Apply the functions to create new 'mass' and 'radius' columns
# df['mass'] = df.apply(calculate_mass, axis=1)
# df['radius'] = df.apply(calculate_radius, axis=1)

# # Drop rows with NaN values in 'mass' or 'radius'
# print(f"Before dropping NaNs in mass/radius: {df.shape}")
# df.dropna(subset=['mass', 'radius'], inplace=True)
# print(f"After dropping NaNs in mass/radius: {df.shape}")

# # Ensure all relevant columns are numeric
# numeric_columns = ['mass_multiplier', 'radius_multiplier', 'distance', 'stellar_magnitude', 'discovery_year', 'mass', 'radius', 'orbital_radius']

# # Print unique values for numeric columns before conversion
# for col in numeric_columns:
#     print(f"Unique values in {col} before conversion:", df[col].unique())

# # Convert columns to numeric and handle errors
# for col in numeric_columns:
#     df[col] = pd.to_numeric(df[col], errors='coerce')

# # Drop rows with NaN values after conversion
# print(f"Before dropping NaNs after conversion: {df.shape}")
# df.dropna(subset=numeric_columns, inplace=True)
# print(f"After dropping NaNs after conversion: {df.shape}")

# # Drop original columns that are no longer needed
# df.drop(['mass_multiplier', 'mass_wrt', 'radius_multiplier', 'radius_wrt'], axis=1, inplace=True)

# # Encode the target variable (planet_type)
# le = LabelEncoder()
# df['planet_type'] = le.fit_transform(df['planet_type'])

# # Prepare features and target
# X = df.drop(['name', 'planet_type'], axis=1)  # Features
# y = df['planet_type']                          # Target variable

# # Convert all feature columns to numeric, forcing errors to NaN
# X = X.apply(pd.to_numeric, errors='coerce')

# # Drop rows with NaN values
# X.dropna(inplace=True)
# y = y[X.index]  # Align y with the cleaned X

# # Check the shape before splitting
# print(f"Shape of features X: {X.shape}, Shape of target y: {y.shape}")

# # Split the dataset into training and testing sets
# if not X.empty and not y.empty:
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Scale the features
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # Train a Random Forest Classifier
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train_scaled, y_train)

#     # Make predictions
#     y_pred = model.predict(X_test_scaled)

#     # Evaluate the model
#     print("Confusion Matrix:")
#     print(confusion_matrix(y_test, y_pred))
#     print("\nClassification Report:")
#     print(classification_report(y_test, y_pred))
# else:
#     print("No data available for training and testing.")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load your dataset
df = pd.read_csv('your_exoplanet_data.csv')

# Display the unique values to understand the data better
print("Unique values in mass_wrt before processing:", df['mass_wrt'].unique())
print("Unique values in radius_wrt before processing:", df['radius_wrt'].unique())

# Ensure there are no NaN values in mass_wrt and radius_wrt
df.dropna(subset=['mass_wrt', 'radius_wrt'], inplace=True)

# Create mappings for mass and radius
mass_mappings = {'Jupiter': 1898, 'Earth': 1}  # mass in Earth masses
radius_mappings = {'Jupiter': 11.2, 'Earth': 1}  # radius in Earth radii

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

# Check shape of DataFrame after processing
print("Shape after processing:", df.shape)

# Define features and target
X = df[['distance', 'stellar_magnitude', 'mass', 'radius', 'orbital_radius']]
y = df['planet_type']  # Assuming 'planet_type' is your target variable

# Check if we have data to split
if X.empty or y.empty:
    print("No data available for training and testing.")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and fit the model
    model = RandomForestClassifier()
    model.fit(X_train_scaled, y_train)

    # Evaluate the model (add your evaluation code here)
    print("Training completed.")
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make predictions on the test set
    y_pred = model.predict(X_test_scaled)

# Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

# Optional: Detailed classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Optional: Confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    import numpy as np

    def predict_planet_type(mass_multiplier, mass_wrt, radius_multiplier, radius_wrt, distance, stellar_magnitude, orbital_radius, discovery_year):
        # Create a mapping for mass_wrt and radius_wrt
        mass_mappings = {'Jupiter': 1898, 'Earth': 1}  # Example mappings
        radius_mappings = {'Jupiter': 69911, 'Earth': 6371}  # Example mappings

        # Calculate mass and radius
        mass = mass_multiplier * mass_mappings[mass_wrt]
        radius = radius_multiplier * radius_mappings[radius_wrt]

        # Prepare the feature array (ensure it matches the training features)
        features = np.array([[mass, radius, distance, stellar_magnitude, orbital_radius]])

        # Scale the features using the same scaler you used for training
        features_scaled = scaler.transform(features)

        # Predict the planet type
        prediction = model.predict(features_scaled)

        return prediction[0]

    # Example usage
    planet_type = predict_planet_type(1.5, 'Earth', 1.2, 'Jupiter', 100, 5, 1.5, 2020)
    print(f"The predicted planet type is: {planet_type}")
