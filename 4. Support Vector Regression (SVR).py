#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


#importing the dataset
dataset = pd.read_csv('C:\\Users\\Dell\\Downloads\\dataset\\Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:,-1].values


# In[4]:


print(x)


# In[5]:


print(y) # need to make this column as vertical so below step


# In[6]:


y = y.reshape(len(y),1)


# In[7]:


print(y)


# In[8]:


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)


# In[9]:


print(x)


# In[10]:


print(y)


# In[11]:


#Training the SVR Model on the whole dataset
from sklearn.svm import SVR
regressor = SVR()
regressor.fit(x,y)


# In[13]:


#  Predicting a new result
sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1,1))


# In[22]:


#visualizing the SVR Results
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y),color='yellow')
plt.plot(sc_x.inverse_transform(x),sc_y.inverse_transform(regressor.predict(x).reshape(-1,1)),color='black')
plt.title('Person Salaries using SVM')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# In[ ]:




