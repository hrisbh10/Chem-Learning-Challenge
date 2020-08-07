import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Loading Training Data
df = pd.read_csv('CLC_train.csv')

#Total Null values in each column
null_val = (df[df == -200].count()*100)/df.shape[0]
for i in range(len(df.columns)):
    print("{} : {}".format(df.columns[i],null_val[i]))

df_wn = df[df != -200]

# #Extracting weekday from dates
import datetime
dys = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
def dateToDay(x):
    dy = datetime.datetime.strptime(x,'%m/%d/%Y').weekday()
    return dys[dy]
df_wn['Days'] = df_wn['Date'].apply(lambda x: dateToDay(x))


# print(df_wn.info())
# print(set(df_wn['CO_level']))
co_dict = {'Very low':0,'Low':1,'Moderate':2,'High':3,'Very High':4}
df_wn['CO_level'] = df_wn['CO_level'].map(co_dict)
# print(df_wn.info())

df_wn.drop(columns=['NMHC_GT'],inplace=True)
df_wn.fillna(value={'CO_GT':0.2,'NO2_GT':50,'Nox_GT':50},inplace=True)
# print(df_wn[df_wn['CO_level']==0].head())

df_won = df_wn.dropna(axis=0)
# print(df_won.info())

# #Plotting the corr matrix
# sns.heatmap(df_won.drop(columns=['Date','Time','CO_level']).corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
# fig=plt.gcf()
# fig.set_size_inches(10,6)
# plt.xticks(rotation=45)
# plt.show()

#Correlation of features with CO level
# corrDF = df_won.drop(columns=['Date','Time','Days']).corrwith(df_won['CO_level'])
# print(corrDF.info())
# plt.bar(corrDF.index,corrDF.loc[:],width=0.2)
# plt.xticks(rotation=40)
# plt.show()

# #Plotting the avg. CO_GT levels of each weekday.
# monday = df_won[(df_won['Days']==0)][['Time','CO_GT']]
# tuesday = df_won[(df_won['Days']==1)][['Time','CO_GT']]
# wednesday = df_won[(df_won['Days']==2)][['Time','CO_GT']]
# thursday = df_won[(df_won['Days']==3)][['Time','CO_GT']]
# friday = df_won[(df_won['Days']==4)][['Time','CO_GT']]
# saturday = df_won[(df_won['Days']==5)][['Time','CO_GT']]
# sunday = df_won[(df_won['Days']==6)][['Time','CO_GT']]

# days = [monday,tuesday,wednesday,thursday,friday,saturday,sunday]

# def plotHours(day):
#     mapp = day.groupby('Time').CO_GT.mean()
#     mapp.index = [x.split(':')[0] for x in mapp.index]
#     mapp.index = [int(x) for x in mapp.index]
#     plt.scatter(mapp.index,mapp,marker='x')

# for day in days:
#     plotHours(day)
# # print(mapp.head())
# plt.legend(dys)

# plt.show()

df_won['Weekend'] = df_won['Days'].apply(lambda x: ((x == dys[-1]) | (x == dys[-2])))
df_won['Weekday'] = df_won['Days'].apply(lambda x: not ((x == dys[-1]) | (x== dys[-2])))
# print(df_won.head())

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = np.array(df_won.drop(columns=['Date','Time','Days','CO_level']))
y = np.array(df_won['CO_level'])
X_train,X_cv,y_train,y_cv = train_test_split(X,y,test_size=0.2)

clf = RandomForestClassifier(n_estimators=20,criterion='gini')
clf.fit(X_train,y_train)
print("Accuracy = {}".format(clf.score(X_cv,y_cv)))
y_pred = clf.predict(X_cv)

from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

print("F1 score = {}".format(f1_score(y_cv,y_pred,average='micro')))
print("Kappa score = {}".format(cohen_kappa_score(y_cv,y_pred)))
print("Confusion Matrix :")
print(confusion_matrix(y_cv,y_pred))






