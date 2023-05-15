#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# Creating the dataset
dataset = pd.read_csv('C:\\Users\\Dell\\Downloads\\dataset\\Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[3]:


print(x)


# In[4]:


print(y)


# In[5]:


# Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)


# In[6]:


print(x_train)


# In[7]:


print(y_train)


# In[9]:


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# In[10]:


print(x_train)


# In[11]:


print(x_test)


# In[12]:


#Training the K-NN Model on the training  dataset
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p=2) #default values
classifier.fit(x_train,y_train)


# In[15]:


#Predicting a  new result
print(classifier.predict(sc.transform([[30,87000]])))


# In[16]:


#Predicting the test set results
y_pred= classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# In[ ]:




