from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the Linear Regression mode
df1 = pd.read_csv('diamonds.csv')

df1 = df1.drop(['Unnamed: 0','color','cut','clarity'],axis=1)

model = LinearRegression()
X = df1.drop(['price','y','z'], axis=1)
y = df1['price']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)
model.fit(X_train,y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the user input from the form
        input_data = [float(request.form['carat']), float(request.form['x']), float(request.form['y']), float(request.form['z'])]

        # Use the model to make a prediction
        prediction = model.predict([input_data])

        # Return the prediction to the user
        return render_template('results.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
