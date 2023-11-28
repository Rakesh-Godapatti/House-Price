#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


boston_dataset = load_boston()
data = pd.DataFrame(data=boston_dataset.data,columns=boston_dataset.feature_names)
data['PRICE']=boston_dataset.target


# In[ ]:


data.head()


# In[ ]:


print('\nNo.Of Entries in DataFrame :',data.shape[0])
print('\n###########################################################################################\n')
#Linear Regression
print('\nFitting Linear Regression............................\n')
features=data.drop(['PRICE'],axis=1)
prices_log=np.log(data['PRICE'])
X_train, X_test, y_train, y_test =train_test_split(features,prices_log,test_size=0.20,random_state=42)
regr=LinearRegression()
regr.fit(X_train,y_train)

Coef=pd.DataFrame(index=X_train.columns,data=regr.coef_,columns=['Coef'])
print(Coef)
print('\nconstant',regr.intercept_)
print('\nTrain_Score(X_train,y_train) :',regr.score(X_train,y_train))
print('\nTest_Score(X_test,y_test) :',regr.score(X_test,y_test))
print('\n#########################################################################################\n')
#stats model-Model_1
print('\nStats Model API......................................\n')
X_incl_const_1=sm.add_constant(X_train)
model_1=sm.OLS(y_train,X_incl_const_1)
results_1=model_1.fit()
print('\nSummary of Model_1 >>>>\n',results_1.summary())
vif_1=[] #empty list
for i in range (X_incl_const_1.shape[1]):
    
    vif_1.append(variance_inflation_factor(exog=X_incl_const_1.values,exog_idx=i))
print('\nVIF_1\n',vif_1)
print('\n#########################################################################################\n')
#Creating Model_2 by Dropping Features from Model_1 Using P_values>0.05
print('\nCreating Model_2 by Dropping Features from Model_1 Using P_values>0.05\n')
X_incl_const_2=sm.add_constant(X_train)
P_values=round(results_1.pvalues,3)
for i in range(P_values.shape[0]):
    if(P_values[i]>0.05):
        X_incl_const_2.drop([P_values.index[i]],axis=1,inplace=True)
model_2=sm.OLS(y_train,X_incl_const_2)
results_2=model_2.fit()
print('\nSummary of Model_2 >>>>\n',results_2.summary())
vif_2=[] #empty list
for i in range (X_incl_const_2.shape[1]):
    
    vif_2.append(variance_inflation_factor(exog=X_incl_const_2.values,exog_idx=i))
print('\nVIF_2\n',vif_2)
print('\n#########################################################################################\n')
#Creating Model_3 by Dropping Features from Model_2 Using VIF>10
print('Creating Model_3 by Dropping Features from Model_2 Using VIF>10')
for i in range(X_incl_const_2.shape[1]):
    if(vif_2[i]>10):
        X_incl_const_2.drop([X_incl_const_2.columns[i]],axis=1,inplace=True)
X_incl_const_3=sm.add_constant(X_incl_const_2)
model_3=sm.OLS(y_train,X_incl_const_3)
results_3=model_3.fit()
print('\nSummary of Model_3 >>>>\n',results_3.summary())
vif_3=[] #empty list
for i in range (X_incl_const_3.shape[1]):
    
    vif_3.append(variance_inflation_factor(exog=X_incl_const_3.values,exog_idx=i))
print('\nVIF_3\n',vif_3)
print('\n#########################################################################################\n')
X_test=X_test[X_incl_const_2.columns]
Pred_y_test=results_3.predict(sm.add_constant(X_test))
print('\nPredicted_Log_Price for X_Test\n',Pred_y_test)
print('\n#########################################################################################\n')
#Getting Required Data 
PT_ratio=round(float(input('\nenter required PT_RATIO :')),1)
Rooms=round(float(input('\nenter required no.of Rooms :')),3)
Chas=input('\nDo Tract should bound Charle\'s River ? [yes/no] : ')
if(Chas=='yes'):
    chas=1.0    
elif(Chas=='no'):
    chas=0.0  
else:
    print('\ndata has both \'yes\' and \'no\' for CHAS')
    chas=data['CHAS']
data_3=X_incl_const_3
X_new=[] # empty list
for i in range (data_3.shape[1]):
    if(data_3.columns[i]=='PTRATIO'):
        X_new.append(PT_ratio)
    elif(data_3.columns[i]=='RM'):
        X_new.append(Rooms)
    elif(data_3.columns[i]=='CHAS'):
        X_new.append(chas)
    else:
        X_new.append(data_3[data_3.columns[i]].mean())
Predicted_Log_Price=results_3.predict(X_new)[0]
print('\nPredicted Price for Your Requirements :',round((30000*np.e**Predicted_Log_Price),3))
upper_bound = 30000*np.e**(Predicted_Log_Price + 2*np.sqrt(results_3.mse_resid))
lower_bound = 30000*np.e**(Predicted_Log_Price - 2*np.sqrt(results_3.mse_resid))
#print('\nRMSE of Model_3 :',round(np.sqrt(results_3.mse_resid),3))
print(f'\nFor Your Requirements Price of a House Ranges From ${round(lower_bound,3)} to ${round(upper_bound,3)}')


# In[ ]:





# In[ ]:




