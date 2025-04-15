from flask import Flask, request, jsonify
from agent import FirePredictionAgent
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the agent
agent = FirePredictionAgent()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['Date', 'Latitude', 'Longitude', 'Humidity', 'Temperature', 'Fire', 'DistanceFromReference']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Convert string values to float
        for field in ['Latitude', 'Longitude', 'Humidity', 'Temperature', 'DistanceFromReference']:
            try:
                data[field] = float(data[field])
            except ValueError:
                return jsonify({'error': f'Invalid value for {field}. Must be a number.'}), 400
        
        # Get prediction
        result = agent.predict(data)
        
        # Return result
        return jsonify({
            'probability': float(result['Fire_Probability'].iloc[0]),
            'prediction': int(result['Fire_Prediction'].iloc[0]),
            'input_data': data
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/train', methods=['POST'])
def train():
    try:
        success = agent.train("processed_data_2023.csv")
        if success:
            return jsonify({'message': 'Model trained successfully'})
        else:
            return jsonify({'error': 'Failed to train model'}), 500
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 