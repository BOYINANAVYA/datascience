#!/usr/bin/env python
# coding: utf-8

# # Simple Linear Regression

# ## Importing the libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[4]:


data=pd.read_csv('Salary_Data.csv')
print(data)


# In[5]:


X=data.iloc[:,:-1]
print(X)


# In[6]:


y=data.iloc[:,-1]
print(data)


# ## Splitting the dataset into the Training set and Test set

# In[8]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/2)
print(X_train)


# In[9]:


print(y_train)


# ## Training the Simple Linear Regression model on the Training set

# In[14]:


from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,y_train)


# In[15]:


print(Regression)


# ## Predicting the Test set results

# In[18]:


y_pred=regression.predict(X_test)
print(y_pred)


# ## Visualising the Training set results

# In[19]:


plt.scatter(X_train,y_train,color='blue')
plt.plot(X_train,regression.predict(X_train),color='red')
plt.tittle("bwkhdhlw")
plt.X_label('kwhk')
plt.y_label('nndkwh')
plt.show()


# ## Visualising the Test set results

# In[20]:


plt.scatter(X_test,y_test,color='blue')
plt.plot(X_test,regression.predict(X_test),color='red')
plt.tittle("bwkhdhlw")
plt.X_label('kwhk')
plt.y_label('nndkwh')
plt.show()


# In[ ]:




