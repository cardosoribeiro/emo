from flask import Flask, jsonify, request
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)


X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])
model = LinearRegression()
model.fit(X, y)

@app.route('/')
def hello_world():
    return jsonify({'message': 'Hello World'})


@app.route('/predict', methods=['POST'])
def predict():
    ''' 
    Predict using the trained linear regression model.
    ---
    tags:
      - prediction
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        description: Input data for prediction.
        required: true
        schema:
          type: object
          properties:
            input:
              type: array
              items:
                type: number
              description: List of numerical values.
              example: [6, 7, 8]
    responses:
      200:
        description: Prediction result.
        schema:
          type: object
          properties:
            prediction:
              type: array
              items:
                type: number
      400:
        description: Bad request.
        schema:
          type: object
          properties:
            error:
              type: string      
    '''
    try:
        data = request.get_json()
        input_data = np.array(data['input']).reshape(-1, 1) 
        prediction = model.predict(input_data).tolist() 
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run()