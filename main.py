from flask import Flask, render_template, request
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import joblib as jb
from sklearn.model_selection import train_test_split
import pandas as pd

app = Flask(__name__)
filename = "Diamond_model.joblib"
model = jb.load(filename)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the user input from the form
        input_data = np.array([float(request.form['carat']), float(request.form['depth']), float(request.form['table']), float(request.form['x'])])
        input_data = np.log(input_data)
        # Use the model to make a prediction
        prediction = model.predict([input_data])
        prediction = np.exp(prediction)
        prediction = np.around(prediction,2)


        # Return the prediction to the user
        return render_template('results.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
