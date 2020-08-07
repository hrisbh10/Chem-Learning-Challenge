import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


labels = {'Very low':0, 'Low':1, 'Moderate':2, 'High':3, 'Very High':4}

traindt = pd.read_csv('traindt.csv')
classes = traindt['CO_level']
trd = traindt.drop(columns=['Date','Time','CO_level','NMHC_GT','Weekday','Weekend'])
classes = np.array([labels[x] for x in classes])

testdt = pd.read_csv('testdt.csv')
class_test = testdt['CO_level']
tsd = testdt.drop(columns=['Date','Time','CO_level','NMHC_GT','Weekday','Weekend'])
class_test = np.array([labels[x] for x in class_test])

y = tf.keras.utils.to_categorical(classes)

data = pd.concat([trd,tsd],axis=0)
means = np.nanmean(data,axis=0)
stddevs = np.nanstd(data,axis=0)

data[data.isna()] = -200
dat_norm = (data - means)/stddevs

x_norm = np.array(dat_norm[:classes.shape[0]])
from sklearn.decomposition import PCA

pca = PCA(n_components=9)
pca.fit(x_norm)
x_norm_pca = pca.transform(x_norm)
x_norm = np.concatenate((x_norm_pca,traindt[['Weekday','Weekend']]),axis=1)
tst_norm = np.array(dat_norm[classes.shape[0]:])
tst_norm_pca = pca.transform(tst_norm)
tst_norm = np.concatenate((tst_norm_pca,testdt[['Weekday','Weekend']]),axis=1)

# print(traindt.info())

# print("Train:",traindt.shape)
print('normalized:',x_norm.shape)
# print("test:",testdt.shape)
# print('normalized:',tst_norm.shape[0])
# y_test = tf.keras.utils.to_categorical(class_test)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

# Xf = scaler.fit_transform(X)
# Xf_test = scaler.transform(X_test)
# lda = 0.1
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(8,input_shape=(x_norm.shape[1],),kernel_regularizer=tf.keras.regularizers.l2(lda),activation='relu'),
#     tf.keras.layers.Dense(8,kernel_regularizer=tf.keras.regularizers.l2(lda),activation='relu'),
#     tf.keras.layers.Dense(5,kernel_regularizer=tf.keras.regularizers.l2(lda+0.02),activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

# model.fit(x_norm,y,epochs=800,batch_size=2048)

# model.save('my_model2')
model = tf.keras.models.load_model('my_model2')
print(model.summary())
ypred_prob = model.predict(tst_norm)
y_pred = np.argmax(ypred_prob,axis=1)
print()
print("Accuracy = {}".format(accuracy_score(class_test,y_pred)))
print("F1 score = {}".format(f1_score(class_test,y_pred,average='micro')))
print("Kappa score = {}".format(cohen_kappa_score(class_test,y_pred)))
print("Confusion Matrix :")
print(confusion_matrix(class_test,y_pred))

