
# coding: utf-8

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from statistics import mode
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from numpy import array
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import f_classif


# In[2]:

#Get data from CSV
print("Loading data from CSV")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
dataSet = pd.read_csv(r'C:\Users\aayus\Desktop\E-Books Uic\Sem2\IML\HW5\responses.csv')
# dataSet.describe()


# In[3]:

# Divide dataset into segments
print("Dividing dataset into segments")
music        = dataSet.iloc[:,0:19]
movies       = dataSet.iloc[:,19:31]
phobias      = dataSet.iloc[:,63:73]
hobbies      = dataSet.iloc[:,31:63]
health       = dataSet.iloc[:,73:76]
personality  = dataSet.iloc[:, 76:133]
demographics = dataSet.iloc[:,140:150]
spending     = dataSet.iloc[:,133:140]


# In[4]:

# Convert string data to numerical categorica data
print("Converting string data to numerical categorica data")
# get unique column values for Punctuality column
personality["Punctuality"].unique()
# convert "Punctuality" column to integers
for i in personality["Punctuality"]:
    if i == "i am often running late":
        personality.replace(i, 1.0, inplace=True)
    elif i == "i am always on time":
        personality.replace(i, 2.0, inplace=True)
    elif i == "i am often early":
        personality.replace(i, 3.0, inplace=True)

personality["Lying"].unique()
# convert "Lying" column to integers
for i in personality["Lying"]:
    if i == "never":
        personality.replace(i, 4.0, inplace=True)
    elif i == "only to avoid hurting someone":
        personality.replace(i, 3.0, inplace=True)
    elif i == "sometimes":
        personality.replace(i, 2.0, inplace=True)
    elif i == "everytime it suits me":
        personality.replace(i, 1.0, inplace=True)

personality["Internet usage"].unique()
# convert "Internet usage" column to integers
for i in personality["Internet usage"]:
    if i == "no time at all":
        personality.replace(i, 4.0, inplace=True)
    elif i == "less than an hour a day":
        personality.replace(i, 3.0, inplace=True)
    elif i == "few hours a day":
        personality.replace(i, 2.0, inplace=True)
    elif i == "most of the day":
        personality.replace(i, 1.0, inplace=True)

health["Smoking"].unique()
# convert "Smoking" column to integers
for i in health["Smoking"]:
    if i == "never smoked":
        health.replace(i, 4.0, inplace=True)
    elif i == "tried smoking":
        health.replace(i, 3.0, inplace=True)
    elif i == "former smoker":
        health.replace(i, 2.0, inplace=True)
    elif i == "current smoker":
        health.replace(i, 1.0, inplace=True)
        
# convert "Alcohol" column to integers
for i in health["Alcohol"]:
    if i == "never":
        health.replace(i, 3.0, inplace=True)
    elif i == "social drinker":
        health.replace(i, 2.0, inplace=True)
    elif i == "drink a lot":
        health.replace(i, 1.0, inplace=True)

demographics["Gender"].unique()
# convert "Gender" column to integers
for i in demographics["Gender"]:
    if i == "female":
        demographics.replace(i, 1.0, inplace=True)
    elif i == "male":
        demographics.replace(i, 2.0, inplace=True)

demographics["Left - right handed"].unique()
# convert "Left - right handed" column to integers
for i in demographics["Left - right handed"]:
    if i == "right handed":
        demographics.replace(i, 1.0, inplace=True)
    elif i == "left handed":
        demographics.replace(i, 2.0, inplace=True)
        
# convert "Education" column to integers
for i in demographics["Education"].unique():
    if i == "currently a primary school pupil":
        demographics.replace(i, 1.0, inplace=True)
    elif i == "primary school":
        demographics.replace(i, 2.0, inplace=True)
    elif i == "secondary school":
        demographics.replace(i, 3.0, inplace=True)
    elif i == "college/bachelor degree":
        demographics.replace(i, 4.0, inplace=True)
    elif i == "masters degree":
        demographics.replace(i, 5.0, inplace=True)
    elif i == "doctorate degree":
        demographics.replace(i, 5.0, inplace=True)
        
demographics["Only child"].unique()
# convert "Only child" column to integers
for i in demographics["Only child"]:
    if i == "yes":
        demographics.replace(i, 1.0, inplace=True)
    elif i == "no":
        demographics.replace(i, 2.0, inplace=True)

demographics["Village - town"].unique()
# convert "Village - Town" column to integers
for i in demographics["Village - town"]:
    if i=="village":
        demographics["Village - town"].replace(i, 1.0, inplace=True)
    elif i=="city":
        demographics["Village - town"].replace(i, 2.0, inplace=True)

demographics["House - block of flats"].unique()
# convert "House - Block of Flats" column to integers
for i in demographics["House - block of flats"]:
    if i == "block of flats":
        demographics["House - block of flats"].replace(i, 1, inplace=True)
    elif i == "house/bungalow":
        demographics["House - block of flats"].replace(i, 2, inplace=True)


# In[5]:

#Feature reduction - convert height and weight to BMI
print("Feature Reduction by converting height and weight to BMI")
hw = list(zip(demographics["Height"], demographics["Weight"]))
bmi = []
for height, weight in hw:
        if (str(height) == "nan" or str(weight) == "nan"):
            bmi.append("nan")
        else:
            result = weight/((height/100)**2)
            bmi.append(float(str(round(result, 2))))

# add "BMI" column to demographics
print("Binning numerical features")
demographics["BMI"] = bmi
# convert BMI to bins
for i in demographics["BMI"]:
    if str(i) != "nan":
        if (15.0 <= i < 16.0):
            demographics["BMI"].replace(i, 1.0, inplace=True)
        elif (16.0 <= i < 18.5):
            demographics["BMI"].replace(i, 2.0, inplace=True)
        elif (18.5 <= i < 25.0):
            demographics["BMI"].replace(i, 3.0, inplace=True)
        elif (25.0 <= i < 30.0):
            demographics["BMI"].replace(i, 4.0, inplace=True)
        elif (i >= 30.0):
            demographics["BMI"].replace(i, 5.0, inplace=True)
    else:
        demographics["BMI"].replace(i, None, inplace=True)
demographics.drop(["Height"], axis=1, inplace=True)
demographics.drop(["Weight"], axis=1, inplace=True)


# In[6]:

# bin "Age" column
for i in demographics["Age"]:
    if (15 <= i < 19):
        demographics["Age"].replace(i, 1.0, inplace=True)
    elif (19 <= i < 23):
        demographics["Age"].replace(i, 2.0, inplace=True)
    elif (23 <= i < 27):
        demographics["Age"].replace(i, 3.0, inplace=True)
    elif (27 <= i < 31):
        demographics["Age"].replace(i, 4.0, inplace=True)
        
# bin "Number of siblings" column
for i in demographics["Number of siblings"]:
    if i == 0:
        demographics["Number of siblings"].replace(i, 0.0, inplace=True)
    elif i == 1:
        demographics["Number of siblings"].replace(i, 1.0, inplace=True)
    elif i == 2:
        demographics["Number of siblings"].replace(i, 2.0, inplace=True)
    elif i == 3:
        demographics["Number of siblings"].replace(i, 3.0, inplace=True)
    elif i == 4:
        demographics["Number of siblings"].replace(i, 4.0, inplace=True)
    elif i > 4:
        demographics["Number of siblings"].replace(i, 5.0, inplace=True)


# In[7]:

#join all segments
print("Joining all segments")
datasets_all = music.join(movies.join(phobias.join(hobbies.join(health.join(personality.join(demographics.join(spending)))))))

#drop all rows containing empty values for Empathy from dataset
datasets_all.dropna(subset=["Empathy"],inplace=True)

#replace na with mean
print("Replacing empty values with mean of columns")
datasets_all = datasets_all.fillna(round(datasets_all.mean()))


# In[8]:

#convert all decimal to int
datasets_all = datasets_all.astype(int)


# In[9]:

y = datasets_all["Empathy"].tolist() #keep y label
datasets_all.drop(["Empathy"], axis=1, inplace=True) #drop empathy from dataset
x = datasets_all.values.tolist()


# In[10]:

print("Finding best parameters using their score value")
#find most frequent y label
most_freq_y=mode(y)
l = [most_freq_y] * len(y)
accuracy = accuracy_score(l, y)
print("Most Frequent Base Classifier Accuracy: %.2f%%" % (accuracy * 100.0))


# In[11]:

#select 40 best features best on score 
print("Dividing dataset into training and test data")
x_new = SelectKBest(chi2, k=40).fit_transform(x, y)
#divide data into 85% training and remaining as testing data
x_train = x_new[:int(len(x_new)*0.85)]
x_test = x_new[int(len(x_new)*0.85):]
y_train = y[:int(len(y)*0.85)]
y_test = y[int(len(y)*0.85):]


# In[12]:

#KNN
print("KNN Started")
neigh = KNeighborsClassifier()
neigh.fit(x_train, y_train)
y_pred = neigh.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("KNN Accuracy: %.2f%%" % (accuracy * 100.0))
scores = cross_val_score(neigh, x_new, y, cv=9)
print("KNN Cross Validation accuracy: %.2f%%" % (scores.mean()*100))


# In[13]:

#Decision Tree
print("Decision Tree Started")
dt = DecisionTreeClassifier(random_state=11)
dt=dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree Accuracy: %.2f%%" % (accuracy * 100.0))
scores = cross_val_score(dt, x_new, y, cv=9)
print("Decision Tree  Cross Validation accuracy: %.2f%%" % (scores.mean()*100))


# In[14]:

#Multinomial Naive Bayes
print("Multinomial Naive Bayes Started")
mnb = MultinomialNB()
mnb=mnb.fit(x_train, y_train)
y_pred = mnb.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Multinomial Naive Bayes Accuracy: %.2f%%" % (accuracy * 100.0))
scores = cross_val_score(mnb, x_new, y, cv=9)
print("Multinomial Naive Bayes Cross Validation accuracy: %.2f%%" % (scores.mean()*100))


# In[15]:

#Gaussian Naive Bayes
print("Gaussian Naive Bayes Started")
gnb = GaussianNB()
scores = cross_val_score(gnb, x_train, y_train, cv=9)
gnb=gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Gaussian Naive Bayes Accuracy: %.2f%%" % (accuracy * 100.0))
scores = cross_val_score(gnb, x_new, y, cv=9)
print("Gaussian Naive Bayes Cross Validation accuracy: %.2f%%" % (scores.mean()*100))


# In[16]:

#Random Forest
print("Random Forest Started")
rf = RandomForestClassifier(max_depth=6, random_state=6)
rf=rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Rando Forest Accuracy: %.2f%%" % (accuracy * 100.0))
scores = cross_val_score(rf, x_new, y, cv=9)
print("Random Forest Cross Validation accuracy: %.2f%%" % (scores.mean()*100))


# In[17]:

#Logistic Regression
print("Logistic Regression Started")
lg = LogisticRegression(multi_class='multinomial', solver='newton-cg')
lg=lg.fit(x_train, y_train)
y_pred = lg.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy: %.2f%%" % (accuracy * 100.0))
scores = cross_val_score(lg, x_new, y, cv=9)
print("Logistic Regression Cross Validation accuracy: %.2f%%" % (scores.mean()*100))


# In[23]:

#XG Boost
print("XGBoost Started")
xgb = XGBClassifier()
xgb=xgb.fit(x_train, y_train)
y_pred = xgb.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("XGBoost Accuracy: %.2f%%" % (accuracy * 100.0))
scores = cross_val_score(xgb, x_new, y, cv=8)
print("XGBoost Cross Validation accuracy: %.2f%%" % (scores.mean()*100))


# In[19]:

#Find best parameters for svm
def svc_param_selection(X, y, nfolds):
    print("Finding best parameters for Kernelized SVM...")
    Cs = [1, 10, 100, 1000]
    gammas = [0.001, 0.01, 0.1, 0.3, 1, 10]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


# In[20]:

#Get best C and Gamma
print("SVM Started")
val = svc_param_selection(x_train, y_train, 9)
c = val.get('C')
gamma = val.get('gamma')


# In[21]:

#RBF Kernalized svm
svmc = svm.SVC(C=c, kernel='rbf', gamma=gamma)
svmc=svmc.fit(x_train, y_train)
y_pred = svmc.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("SVM RBF Kernel Accuracy: %.2f%%" % (accuracy * 100.0))
scores = cross_val_score(svmc, x_new, y, cv=9)
print("SVM RBF Kernel Cross Validation accuracy: %.2f%%" % (scores.mean()*100))

