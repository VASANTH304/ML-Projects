#!/usr/bin/env python
# coding: utf-8

# In[5]:


#importing the modules 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[6]:


#importing the dataset
dataset = pd.read_csv('C:\\Users\\Dell\\Downloads\\dataset\\Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# In[8]:


# splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=1)


# In[10]:


#Training the simple linear regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[11]:


#predicting the test set results
y_pred = regressor.predict(x_test)


# In[14]:


#visualising the training set module
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='black')
plt.title('salary vs Experience for employees (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[15]:


#visualising the test set module
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='black')
plt.title('Salary vs Experience for employees (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:




