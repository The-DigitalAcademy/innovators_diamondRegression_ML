from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the Linear Regression mode
df = pd.read_csv('diamonds.csv')
df = df[(df.z != 0) & (df.x != 0) & (df.y != 0)]
df1 = df[df.y < 30 ]
df1 = df1[df1.z < 30 ]
df1 = df1.drop(['Unnamed: 0','color','cut','clarity'],axis=1)
df1 = np.log(df1)

model = LinearRegression()
X = df1.drop(['price','y','z'], axis=1)
y = df1['price']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=49)
model.fit(X_train,y_train)

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
        
        
        r2 = r2_score(y_train, model.predict(X_train))
        skew = df1['price'].skew()
        kurt = df1['price'].kurt()

        # Return the prediction to the user
        return render_template('results.html', prediction=prediction[0],r2 = r2,skew = skew, kurt = kurt)

if __name__ == '__main__':
    app.run(debug=True)
