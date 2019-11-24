#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


dataset = pd.read_csv(r"C:\Users\BISWA\Desktop\ML Project\Purchased_Dataset.csv")


# In[3]:


dataset.head()


# In[4]:


dataset.columns


# In[5]:


x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
x_train,x_test,y_train,y_test = train_test_split(x,y, random_state=42)
Classifier = KNeighborsClassifier(n_neighbors=5)
Classifier.fit(x_train,y_train)
y_pre = Classifier.predict(x_test)
accuracy_score(y_test,y_pre)


# In[18]:


from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
Classifier = KNeighborsClassifier(n_neighbors=4)
print(cross_val_score(Classifier,x,y, cv=10,scoring='accuracy').mean())


# In[21]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
print(cross_val_score(model,x,y, cv=10,scoring='accuracy').mean())


# In[25]:


from sklearn.svm import SVC
svcmodel = SVC()
print(cross_val_score(svcmodel,x,y, cv=10, scoring="accuracy").mean())


# In[26]:


from sklearn.naive_bayes import GaussianNB
nbmodel = GaussianNB()
print(cross_val_score (nbmodel,x,y, cv=10,scoring='accuracy').mean())


# # Different type of classifier algorthim accuracy
# 1. KNeighborsClassifier = 78%
# 2. LogisticRegression = 64%
# 3. SVC = 68%
# 4. naive_bayes = 87%
# 
# 
