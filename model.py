import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras import layers, models

# Load your dataset (replace 'your_dataset.csv' with your actual dataset)
df = pd.read_csv('Dataset.csv')

# Sample features (customize based on your dataset)
features = df[['age', 'gender', 'activity_level', 'weight(kg)', 'height(m)', 'BMI', 'BMR']]
labels = df['calories_to_maintain_weight']

# Preprocess data
label_encoder = LabelEncoder()
features['gender'] = label_encoder.fit_transform(features['gender'])
features['activity_level'] = label_encoder.fit_transform(features['activity_level'])

# Normalize numerical features
scaler = StandardScaler()
features[['age', 'weight(kg)', 'height(m)', 'BMI', 'BMR']] = scaler.fit_transform(features[['age', 'weight(kg)', 'height(m)', 'BMI', 'BMR']])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Build the model
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(features.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Adjust the output size based on your target variable
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f'Test MAE: {test_mae}')
# Save the model
model.save('calories_to_maintain_weight4.h5')
