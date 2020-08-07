import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

def data_process(df):
    df_wn = df[df != -200]

    # #Extracting weekday from dates
    import datetime
    dys = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    def dateToDay(x):
        dy = datetime.datetime.strptime(x,'%m/%d/%Y').weekday()
        return dys[dy]
    df_wn['Days'] = df_wn['Date'].apply(lambda x: dateToDay(x))

    co_dict = {'Very low':0,'Low':1,'Moderate':2,'High':3,'Very High':4}
    df_wn['CO_level'] = df_wn['CO_level'].map(co_dict)

    df_wn.drop(columns=['NMHC_GT'],inplace=True)
    df_wn.fillna(value={'CO_GT':0,'NO2_GT':50,'Nox_GT':50},inplace=True)
    df_won = df_wn.dropna(axis=0)

    df_won['Weekend'] = df_won['Days'].apply(lambda x: ((x == dys[-1]) | (x == dys[-2])))
    df_won['Weekday'] = df_won['Days'].apply(lambda x: not ((x == dys[-1]) | (x== dys[-2])))

    return df_won

def generate_scores(X,y,clf):
    print("Accuracy = {}".format(clf.score(X,y)))
    y_pred = clf.predict(X)

    print("F1 score = {}".format(f1_score(y,y_pred,average='micro')))
    print("Kappa score = {}".format(cohen_kappa_score(y,y_pred)))
    print("Confusion Matrix :")
    print(confusion_matrix(y,y_pred))