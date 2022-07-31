#!/usr/bin/env python
# coding: utf-8

# # Problem statement

# The Goal is to predict whether a person has an income of more than 50K a year or not.
# This is basically a binary classification problem where a person is classified into the
# >50K group or <=50K group.

# Prediction task is to determine whether a person makes over 50K a year.

# # Importing necessary libraries

# In[1]:


import numpy as np
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# # Getting data

# In[2]:


df= pd.read_csv("C:/Users/Narashima Rao/Documents/Ineuron Dashboard Pro/machine-learning/adult_income_prediction/adult.csv")


# In[3]:


columns = ['Age','Workclass','fnlgwt','Education','Education num','Marital Status',
           'Occupation','Relationship','Race','Sex','Capital Gain','Capital Loss',
           'Hours/Week','Native country','Income']


# In[4]:


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

# In[5]:


df.head(10)


# In[6]:


df.info()


# In[8]:


# Shape of data
print('Rows: {} Columns: {}'.format(df.shape[0], df.shape[1]))


# In[10]:


df.shape


# In[11]:


#Statistical summary
df.describe()


# In[12]:


# Checking null values in data
round((df.isnull().sum() / df.shape[0]) * 100, 2).astype(str) + ' %'


# In[13]:


# Checking for '?' in data
round((df.isin(['?']).sum() / df.shape[0])
      * 100, 2).astype(str) + ' %'


# In[15]:


# Checking the counts of income category
salary = df['salary'].value_counts(normalize=True)
round(salary * 100, 2).astype('str') + ' %'


# # Observations

# The given data doesn't have any null values,and didn't contain missing values in the form of '?' which needs to be preprocessed.
# 
# The data is unbalanced, as the dependent feature 'salary' contains 75.92% values have income less than 50k and 24.08% values have income more than 50k.

# In[ ]:





# # Exploratory Data Analysis

# ## Univariate Analysis

# In[17]:


# barplot for 'Income'
salary = df['salary'].value_counts()

plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(7, 5))
sns.barplot(salary.index, salary.values, palette='bright')
plt.title('Distribution of Income', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Income', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=10)
plt.show()


# In[19]:


# distribution plot for 'Age'
age = df['age'].value_counts()

plt.figure(figsize=(10, 5))
plt.style.use('fivethirtyeight')
sns.distplot(df['age'], bins=20)
plt.title('Distribution of Age', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Age', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=10)
plt.show()


# In[20]:


#barplot for 'Education'
education = df['education'].value_counts()

plt.style.use('seaborn')
plt.figure(figsize=(10, 5))
sns.barplot(education.values, education.index, palette='Paired')
plt.title('Distribution of Education', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Education', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.show()


# In[22]:


df.columns


# In[23]:


#barplot for 'Years of Education'
edu_num = df['education-num'].value_counts()

plt.style.use('ggplot')
plt.figure(figsize=(10, 5))
sns.barplot(edu_num.index, edu_num.values, palette='colorblind')
plt.title('Distribution of Years of Education', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Years of Education', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.show()


# In[24]:


# Create pie chart for 'Marital status'
marital = df['marital-status'].value_counts()

plt.style.use('default')
plt.figure(figsize=(10, 7))
plt.pie(marital.values, labels=marital.index, startangle=10, explode=(
    0, 0.20, 0, 0, 0, 0, 0), shadow=True, autopct='%1.1f%%')
plt.title('Marital distribution', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.legend()
plt.legend(prop={'size': 7})
plt.axis('equal')
plt.show()


# In[26]:


# donut chart for 'relationship'
relationship = df['relationship'].value_counts()

plt.style.use('bmh')
plt.figure(figsize=(20, 10))
plt.pie(relationship.values, labels=relationship.index,
        startangle=50, autopct='%1.1f%%')
centre_circle = plt.Circle((0, 0), 0.7, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title('Relationship distribution', fontdict={
          'fontname': 'Monospace', 'fontsize': 30, 'fontweight': 'bold'})
plt.axis('equal')
plt.legend(prop={'size': 15})
plt.show()


# In[28]:


#barplot for 'Sex'
gender = df['sex'].value_counts()

plt.style.use('default')
plt.figure(figsize=(7, 5))
sns.barplot(gender.index, gender.values)
plt.title('Distribution of Sex', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Sex', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=10)
plt.grid()
plt.show()


# In[30]:


import squarify


# In[31]:


# A Treemap for 'Race'
race = df['race'].value_counts()

plt.style.use('default')
plt.figure(figsize=(7, 5))
squarify.plot(sizes=race.values, label=race.index, value=race.values)
plt.title('Race distribution', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.show()


# In[32]:


#barplot for 'Hours per week'
hours = df['hours-per-week'].value_counts().head(10)

plt.style.use('bmh')
plt.figure(figsize=(15, 7))
sns.barplot(hours.index, hours.values, palette='colorblind')
plt.title('Distribution of Hours of work per week', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Hours of work', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.show()


# ## Bivariate Analysis

# In[35]:


#countplot of income across age
plt.style.use('default')
plt.figure(figsize=(20, 7))
sns.countplot(df['age'], hue=df['salary'])
plt.title('Distribution of Salary across Age', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Age', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.legend(loc=1, prop={'size': 15})
plt.show()


# In[34]:


#countplot of income across education
plt.style.use('seaborn')
plt.figure(figsize=(20, 7))
sns.countplot(df['education'],
              hue=df['salary'], palette='colorblind')
plt.title('Distribution of Salary across Education', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Education', fontdict={'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.legend(loc=1, prop={'size': 15})
plt.show()


# In[39]:


# countplot of salary across years of education
plt.style.use('bmh')
plt.figure(figsize=(20, 7))
sns.countplot(df['education-num'],
              hue=df['salary'])
plt.title('Salary across Years of Education', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Years of Education', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.legend(loc=1, prop={'size': 15})
plt.savefig('bi2.png')
plt.show()


# In[41]:


# countplot of salary across Marital Status
plt.style.use('seaborn')
plt.figure(figsize=(20, 7))
sns.countplot(df['marital-status'], hue=df['salary'])
plt.title('Salary across Marital Status', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Marital Status', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.legend(loc=1, prop={'size': 15})
plt.show()


# In[43]:


# countplot of Salary across race
plt.style.use('fivethirtyeight')
plt.figure(figsize=(20, 7))
sns.countplot(df['race'], hue=df['salary'])
plt.title('Distribution of Salary across race', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Race', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.legend(loc=1, prop={'size': 15})
plt.show()


# In[45]:


#countplot of salary across sex
plt.style.use('fivethirtyeight')
plt.figure(figsize=(7, 3))
sns.countplot(df['sex'], hue=df['salary'])
plt.title('Distribution of salary across sex', fontdict={
          'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
plt.xlabel('Sex', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.ylabel('Number of people', fontdict={
           'fontname': 'Monospace', 'fontsize': 15})
plt.tick_params(labelsize=12)
plt.legend(loc=1, prop={'size': 10})
plt.savefig('bi3.png')
plt.show()


# ## Multivariate Analysis

# In[46]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[47]:


df['salary'] = le.fit_transform(df['salary'])


# In[48]:


# Creating a pairplot of dataset
sns.pairplot(df)
plt.savefig('salary_multi1.png')
plt.show()


# In[50]:


corr = df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True,
                     annot=True, cmap='RdYlGn')
plt.savefig('salary_multi2.png')
plt.show()


# ## Find correlation between columns

# In[11]:


def correlation(df, size=15):
    corr= df.corr()
    fig, ax =plt.subplots(figsize=(size,size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)),corr.columns)
    plt.yticks(range(len(corr.columns)),corr.columns)
    plt.show()


# In[12]:


correlation(df)


# ## Categorize US and Non-US people

# In[13]:


print (df[['country','salary']].groupby(['country']).mean())


# # Data visualization

# In[14]:


fig = plt.figure(figsize=(10,6))

sns.countplot('workclass', data=df)
plt.tight_layout()
plt.show()


# In[15]:


fig = plt.figure(figsize=(20,6))

sns.countplot('education', data=df)
plt.tight_layout()
plt.show()


# In[16]:


fig = plt.figure(figsize=(20,6))

sns.countplot('marital-status', data=df)
plt.tight_layout()
plt.show()


# In[17]:


fig = plt.figure(figsize=(20,6))

sns.countplot('occupation', data=df)
plt.tight_layout()
plt.show()


# In[18]:


temp= df
hmap = temp.corr()
plt.subplots(figsize=(12, 12))
sns.heatmap(hmap, vmax=.8,annot=True,cmap="BrBG", square=True);


# # Conclusion

# In this dataset, the most number of people are young, white, male, high school graduates with 9 to 10 years of education and work 40 hours per week.
# 
# From the correlation heatmap, we can see that the dependent feature 'salary' is highly correlated with age, numbers of years of education, capital gain and number of hours per week.

# In[ ]:




