#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


# Creating the dataset 
dataset = pd.read_csv('C:\\Users\\Dell\\Downloads\\dataset\\Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values


# In[5]:


print(x)


# In[6]:


print(y)


# In[38]:


#Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)


# In[39]:


print(x_train)


# In[40]:


print(x_test)


# In[41]:


print(y_train)


# In[42]:


print(y_test)


# In[43]:


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc =  StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# In[44]:


print(x_train)


# In[45]:


print(x_test)


# In[46]:


# Training the SVM Model on the Training dataset
from sklearn.svm import SVC
classifier = SVC(kernel='linear',random_state=0)
classifier.fit(x_train,y_train)


# In[47]:


#predicting a new result
print(classifier.predict(sc.transform([[30,87000]])))


# In[48]:


y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[ ]:




