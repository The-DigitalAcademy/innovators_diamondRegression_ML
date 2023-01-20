import joblib as jb
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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

filename = "Diamond_model.joblib"
jb.dump(model, filename)
print("model saved successfully!")