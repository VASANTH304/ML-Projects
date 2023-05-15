#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


#importing the dataset
dataset = pd.read_csv('C:\\Users\\Dell\\Downloads\\dataset\\Social_Network_Ads.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# In[3]:


print(x)


# In[4]:


print(y)


# In[5]:


# Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test , y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)


# In[6]:


print(x_train)


# In[7]:


print(y_train)


# In[8]:


print(x_test)


# In[9]:


print(y_test)


# In[12]:


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# In[13]:


print(x_train)


# In[14]:


print(x_test)


# In[15]:


# Training the KERNEL SVM Model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0) # rbf deines non linear
classifier.fit(x_train,y_train)


# In[16]:


#Predicting a new result
print(classifier.predict(sc.transform([[30,87000]])))


# In[17]:


#Predicting the test set results
y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1) , y_test.reshape(len(y_test),1)),1))


# In[ ]:




