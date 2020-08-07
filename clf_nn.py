import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

train_dt = pd.read_csv('train_ready.csv')
test_dt = pd.read_csv('test_ready.csv')

X = np.array(train_dt.drop(columns=['Date','Time','CO_level']))
X_test = np.array(test_dt.drop(columns=['Date','Time','CO_level']))

labels = {'Very low':0, 'Low':1, 'Moderate':2, 'High':3, 'Very High':4}
classes = np.array([labels[x] for x in train_dt['CO_level']])
class_test = np.array([labels[x] for x in test_dt['CO_level']])

y = tf.keras.utils.to_categorical(classes)
# y_test = tf.keras.utils.to_categorical(class_test)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

Xf = scaler.fit_transform(X)
Xf_test = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(12,input_shape=(X.shape[1],),activation='relu'),
    tf.keras.layers.Dense(7,activation='relu'),
    tf.keras.layers.Dense(5,activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(Xf,y,epochs=200,batch_size=1024)

model.save('my_model')

ypred_prob = model.predict(Xf_test)
y_pred = np.argmax(ypred_prob,axis=1)
print()
print("Accuracy = {}".format(accuracy_score(class_test,y_pred)))
print("F1 score = {}".format(f1_score(class_test,y_pred,average='micro')))
print("Kappa score = {}".format(cohen_kappa_score(class_test,y_pred)))
print("Confusion Matrix :")
print(confusion_matrix(class_test,y_pred))

