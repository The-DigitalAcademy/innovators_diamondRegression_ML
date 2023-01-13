from flask import Flask, render_template, request
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import joblib as jb
from sklearn.model_selection import train_test_split
import pandas as pd

app = Flask(__name__)
filename = "Diamond_model.joblib"
model = jb.load(filename)


# Load the Linear Regression mode
df = pd.read_csv('diamonds.csv')
df = df[(df.z != 0) & (df.x != 0) & (df.y != 0)]
df1 = df[df.y < 30 ]
df1 = df1[df1.z < 30 ]
df1 = df1.drop(['Unnamed: 0','color','cut','clarity'],axis=1)
df1 = np.log(df1)

X = df1.drop(['price','y','z'], axis=1)
y = df1['price']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=49)

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

        mse_train = mean_squared_error(y_train, model.predict(X_train))
        mse_train = np.exp(mse_train)
        mse_train = np.around(mse_train,2)
        mse_test = mean_squared_error(y_test, model.predict(X_test))
        mse_test = np.exp(mse_test)
        mse_test = np.around(mse_test,2)

        actual = prediction in np.asarray(df1)
    
        r2 = r2_score(y_train, model.predict(X_train))
        r2 = float(r2)
        r2 = r2*100
        r2 = round(r2,2)

        # Return the prediction to the user
        return render_template('results.html', prediction=prediction[0],actual=actual,r2 = r2,mse_train=mse_train,mse_test=mse_test)

if __name__ == '__main__':
    app.run(debug=True)
