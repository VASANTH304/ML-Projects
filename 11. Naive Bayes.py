#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


# Importing the dataset
dataset = pd.read_csv('C:\\Users\\Dell\\Downloads\\dataset\\Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[4]:


print(x)


# In[5]:


print(y)


# In[6]:


# Splitting the dataset onto the training and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)


# In[7]:


print(x_train)


# In[8]:


print(y_train)


# In[9]:


print(x_test)


# In[10]:


print(y_test)


# In[12]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# In[13]:


print(x_train)


# In[14]:


print(x_test)


# In[16]:


# Training the Naive Bayes model on the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)


# In[17]:


# Predicting the new result
print(classifier.predict(sc.transform([[30,87000]])))


# In[18]:


#predicting the test set results
y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1) , y_test.reshape(len(y_test),1)),1))


# In[19]:


#Making the confusion matrix
from sklearn.metrics import confusion_matrix , accuracy_score
print(confusion_matrix(y_test, y_pred))
accuracy_score(y_test, y_pred)


# In[ ]:




