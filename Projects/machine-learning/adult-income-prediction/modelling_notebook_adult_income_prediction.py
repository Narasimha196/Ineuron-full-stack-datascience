#!/usr/bin/env python
# coding: utf-8

# # Problem statement

# The Goal is to predict whether a person has an income of more than 50K a year or not.
# This is basically a binary classification problem where a person is classified into the
# >50K group or <=50K group.

# Prediction task is to determine whether a person makes over 50K a year.

# # Importing libraries

# In[49]:


import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# # Getting data

# In[4]:


df= pd.read_csv("C:/Users/Narashima Rao/Documents/Ineuron Dashboard Pro/machine-learning/adult_income_prediction/adult.csv")


# In[5]:


columns = ['Age','Workclass','fnlgwt','Education','Education num','Marital Status',
           'Occupation','Relationship','Race','Sex','Capital Gain','Capital Loss',
           'Hours/Week','Native country','Income']


# In[6]:


df.columns


# # Attribute Information:

# Listing of attributes:
# 
# >50K, <=50K.
# 
# age: continuous.
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# 
# fnlwgt: continuous.
# 
# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# 
# education-num: continuous.
# 
# marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# 
# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# 
# relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# 
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# 
# sex: Female, Male.
# 
# capital-gain: continuous.
# 
# capital-loss: continuous.
# 
# hours-per-week: continuous.
# 
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan,
# 
# Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, 
# 
# Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, 
# 
# El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

# # Data Exploration

# In[7]:


df.head(10)


# In[8]:


df.info()


# In[9]:


# Shape of data
print('Rows: {} Columns: {}'.format(df.shape[0], df.shape[1]))


# In[10]:


df.shape


# # Data cleaning

# ## Fixing '?' values in the dataset

# In[11]:


df = df.replace('?', np.nan)


# In[12]:


# null values in the data
round((df.isnull().sum() / df.shape[0]) * 100, 2).astype(str) + ' %'


# There are no null values in the dataset

# # Label Encoding

# In[13]:


for col in df.columns:
    if df[col].dtypes == 'object':
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])


# # Feature Selection

# In[14]:


X = df.drop('salary', axis=1)
y = df['salary']


# In[15]:


etc_selector = ExtraTreesClassifier(random_state=42)


# In[16]:


etc_selector.fit(X, y)


# In[17]:


feature_imp = etc_selector.feature_importances_


# In[18]:


for index, val in enumerate(feature_imp):
    print(index, round((val * 100), 2))


# In[19]:


X.info()


# In[20]:


X = X.drop(['workclass', 'education', 'race', 'sex',
            'capital-loss', 'country'], axis=1)


# # Feature Scaling

# In[21]:


for col in X.columns:
    scaler = StandardScaler()
    X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))


# In[88]:


print(X[col])


# # Fixing imbalanced data using Oversampling technique

# In[22]:


round(y.value_counts(normalize=True) * 100, 2).astype('str') + ' %'


# In[23]:


over_sampler = RandomOverSampler(random_state=42)


# In[24]:


over_sampler.fit(X,y)


# In[25]:


X_over, y_over = over_sampler.fit_resample(X, y)


# In[26]:


round(y_over.value_counts(normalize=True) * 100, 2).astype('str') + ' %'


# # Train Test Split

# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2, random_state=42)


# In[30]:


print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", y_train.shape)
print("Y_test shape:", y_test.shape)


# # Models

# 1. Logistic Regression
# 
# 2. Naive baye's
# 
# 3. KNN
# 
# 4. Decision tree classifier
# 
# 5. Random forest classifier
# 
# 6. Support Vector Classifier(SVC)
# 
# 7. Xgboost classifier

# In[42]:


log = LogisticRegression(random_state=42)
nb = GaussianNB()
knn = KNeighborsClassifier()
dtc = DecisionTreeClassifier(random_state=42)
rfc = RandomForestClassifier(random_state=42)
svc = SVC(random_state=42)
xgb = XGBClassifier()


# In[40]:


def build_model(model,X_train,y_train,X_test,y_test):
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    clf_report = classification_report(y_test,y_pred)
    print(f"The accuracy of the model {type(model).__name__} is {accuracy :.2f}")
    print(confusion_matrix(y_test,y_pred))
    print(clf_report)
    print("\n")


# In[ ]:





# ## Logistic Regression

# In[41]:


build_model(log,X_train,y_train,X_test,y_test)


# ## Naive baye's

# In[43]:


build_model(nb,X_train,y_train,X_test,y_test)


# ## KNN

# In[44]:


build_model(knn,X_train,y_train,X_test,y_test)


# ## Decision tree classifier

# In[45]:


build_model(dtc,X_train,y_train,X_test,y_test)


# ## Random forest classifier

# In[46]:


build_model(rfc,X_train,y_train,X_test,y_test)


# ## SVC

# In[47]:


build_model(svc,X_train,y_train,X_test,y_test)


# ## Xgboost classifier

# In[48]:


build_model(xgb,X_train,y_train,X_test,y_test)


# # Hyperparameter Tuning

# In[50]:


n_estimators = [int(x) for x in np.linspace(start=40, stop=150, num=15)]
max_depth = [int(x) for x in np.linspace(40, 150, num=15)]


# In[55]:


params_dist = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
}


# In[53]:


rfc_tuned = RandomForestClassifier(random_state=42)


# In[56]:


rfc_cv = RandomizedSearchCV(
    estimator=rfc_tuned, param_distributions=params_dist, cv=5, random_state=42)


# In[73]:


rfc_cv.fit(X_train, y_train)


# In[74]:


rfc_cv.score


# In[77]:


rfc_best = RandomForestClassifier(max_depth=102, n_estimators=40, random_state=42)


# In[79]:


rfc_best.fit(X_train, y_train)


# In[82]:


y_pred_rfc = rfc_best.predict(X_test)


# In[83]:


print('Random Forest Classifier:')
print('Accuracy score:', round(accuracy_score(y_test, y_pred_rfc) * 100, 2))
print('F1 score:', round(f1_score(y_test, Y_pred_rfc) * 100, 2))


# In[86]:


cm = confusion_matrix(y_test, y_pred_rfc)
print(cm)
clf_report = classification_report(y_test, y_pred_rfc)
print(clf_report)


# In[87]:


plt.style.use('default')
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
plt.savefig('heatmap.png')
plt.show()


# # Conclusion

# In this project, we build various models like logistic regression, knn classifier, support vector classifier, decision tree classifier, random forest classifier and xgboost classifier.
# 
# A hyperparameter tuned random forest classifier gives the same accuracy score of 93% and f1 score of 93%.

# # Future scope

# We have a large enough dataset, so we can use neural networks such as an artificial neural network to build a model which can result in better performance.

# In[ ]:




