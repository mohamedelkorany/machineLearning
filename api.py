from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the pre-trained machine learning model
model = load_model('calories_to_maintain_weight.h5')

# Function to preprocess input data before making predictions
def preprocess_data(data):
    # Implement your preprocessing logic here
    # Example: convert categorical variables, normalize numerical features, etc.
    return np.array(data).reshape(1, -1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        input_data = request.json['data']

        # Preprocess the input data
        processed_data = preprocess_data(input_data)

        # Make predictions using the loaded model
        predictions = model.predict(processed_data)

        # You might need to post-process predictions based on your model output

        # Send predictions back to the Flutter app
        response = {'predictions': predictions.tolist()}

        def calCarbs(calories):
            carbs = (calories * (55/100)) / 4
            return carbs

        def calProtein(calories):
            protein = (calories * (25/100)) / 4
            return protein

        def calFats(calories):
            fat = (calories * (20/100)) / 9
            return fat
        
        
        print(response['predictions'])
        caloriesValue = response['predictions'][0][0]
        carbsValue = calCarbs(caloriesValue)
        proteinValue = calProtein(caloriesValue)
        fatsValue = calFats(caloriesValue)
        responseData = {'calories': caloriesValue,'carbs': carbsValue, 'protein': proteinValue, 'fat': fatsValue}
        return jsonify(responseData)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)