
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer as ct
from sklearn.preprocessing import LabelEncoder as le, OneHotEncoder as ohe
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as sc
from sklearn.impute import KNNImputer as kni

import keras
from keras.models import Sequential as sq
from keras.layers import Dense as ds

def train_model(layers, neurons):
    
    model = sq()
    model.add(ds(neurons, kernel_initializer='uniform', activation='relu', input_shape=(X_train.shape[1],)))
    
    for _ in range(layers-1):
        model.add(ds(neurons, kernel_initializer='uniform', activation='relu'))
        
    model.add(ds(1, kernel_initializer='uniform', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=300, batch_size=32)
    
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    train_loss, train_acc = model.evaluate(X_train, Y_train)
    return test_acc , train_acc
   
df = pd.read_csv('bullet_train.csv')
df["Age"] = kni(n_neighbors=5).fit_transform(df[["Age"]])
df[["Age", "Fare"]] = sc().fit_transform(df[["Age", "Fare"]])
X = df.iloc[:,1:].values
Y = df.iloc[:,0].values.astype('float32')


pclass = ct([("Pclass", ohe(), [0])], remainder = 'passthrough')
X[:,1] = le().fit_transform(X[:,1])

X = pclass.fit_transform(X)
X = X[:,1:].astype('float32')

X_train, X_test, Y_train, Y_test = tts(X, Y, test_size=0.2, random_state=0)

test_accuracies = []
for layers in range(3, 6): 
  for neurons in range(6, 64, 10):
    test_acc , train_acc = train_model(layers, neurons)
    test_accuracies.append((layers, neurons, test_acc, train_acc))
    
best_model = max(test_accuracies, key=lambda X: X[2])
layers, neurons, best_test_acc, train_acc = best_model

