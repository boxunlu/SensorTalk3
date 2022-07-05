#!/usr/bin/env python
# coding: utf-8

# In[36]:


## Import requirements packages for SensorTalk3.
import numpy as np 
import joblib
import pandas as pd
import warnings
import xgboost as xgb
from sklearn import  ensemble
from sklearn.linear_model import LinearRegression


# In[37]:


def cal_mape(y_ture, y_pred):
    return np.mean(np.abs((y_ture - y_pred) / y_ture)) * 100

def systematic_sampling(df, step):
 
    indexes = np.arange(0, len(df), step=step)
    systematic_sample = df.iloc[indexes]
    df = df.drop(indexes)
    return systematic_sample, df

def EnsembleMethod(STD_EC, STD_EC_Label, DUT_Validate, DUT_Validate_Label, DUT, DUT_Label):
       
    XGB_Model = xgb.XGBRegressor(n_estimators = 100, 
                                 max_depth = 6, 
                                 eta = 0.3, 
                                 gamma = 0) 
    XGB_Model.fit(STD_EC,STD_EC_Label)
    
    XGB_pred_train = XGB_Model.predict(DUT) ## 用測試集的結果
    XGB_pred_result = XGB_Model.predict(DUT_Validate) 
    
    #----------------------------------------------------------------------------------------------#
    RF_Model = ensemble.RandomForestRegressor(n_estimators = 100, 
                                              max_depth = None, 
                                              min_samples_split = 2,
                                              max_samples = 1
                                              ) 
    
    RF_Model.fit(STD_EC,STD_EC_Label)
    
    RF_pred_train = RF_Model.predict(DUT) ## 用測試集的結果
    RF_pred_result = RF_Model.predict(DUT_Validate)    
    #----------------------------------------------------------------------------------------------#
    
    Ensemble_model = LinearRegression(fit_intercept=True,copy_X=True)

    F1 = pd.DataFrame(DUT_Label)
    F2 = pd.DataFrame(RF_pred_train)
    F3 = pd.DataFrame(XGB_pred_train)
    Ensemble_Dataset = pd.concat([F1,F2,F3],axis = 1).values
    Ensemble_Dataset = np.array(Ensemble_Dataset, dtype=np.float64)

    Ensemble_EC = Ensemble_Dataset[:,1:]
    Ensemble_EC_label = Ensemble_Dataset[:,0]
    
    
    val_F1 = pd.DataFrame(DUT_Validate_Label)
    val_F2 = pd.DataFrame(RF_pred_result)
    val_F3 = pd.DataFrame(XGB_pred_result)

    Ensemble_Dataset_validate = pd.concat([val_F1,val_F2,val_F3],axis = 1).values
    Ensemble_EC_Validate = Ensemble_Dataset_validate[:,1:]
    Ensemble_EC_Validate_label = Ensemble_Dataset_validate[:,0]
    
    Ensemble_model.fit(Ensemble_EC,Ensemble_EC_label)
    Ensemble_result = Ensemble_model.predict(Ensemble_EC_Validate)
    
    return Ensemble_result


# In[38]:


## Load Dataset

df_train = pd.read_excel("./Dataset1.xlsx")
df_train = df_train.reset_index(drop=True)

#--------------TrainData----------------------#
TrainingDataset = df_train.iloc[:]
TestingDataset,TrainingDataset = systematic_sampling(TrainingDataset,2)

#--------------Validation----------------------#
ValidationDataset = pd.read_excel("./Dataset3.xlsx")
ValidationDataset = ValidationDataset.iloc[:]

TrainingDataset = np.array(TrainingDataset, dtype=np.float64)
TestingDataset = np.array(TestingDataset, dtype=np.float64)
ValidationDataset = np.array(ValidationDataset, dtype=np.float64)

STD_EC = TrainingDataset[:,1:6] ## 0是標準(Label)、1是待校正。 #建表用
STD_EC_Label = TrainingDataset[:,0] ## STD_Label。

DUT = TestingDataset[:,1:] ## 測試集 feature
DUT_Label = TestingDataset[:,0] ## 測試集 Label

DUT_Validate = ValidationDataset[:,1:6] ## 驗證集 feature
DUT_Validate_Label = ValidationDataset[:,0] ## 驗證集 Label


# In[39]:


## The MAPE of the original dataset.
Original_MAPE = cal_mape(DUT_Validate_Label,DUT_Validate[:,0])

Ensemble_MAPE_list = []
FinalOutput = EnsembleMethod(STD_EC, STD_EC_Label, DUT_Validate, DUT_Validate_Label, DUT,DUT_Label)
Ensemble_MAPE = cal_mape(DUT_Validate_Label, FinalOutput)
Ensemble_MAPE_list.append(Ensemble_MAPE)


# In[40]:


print('Original MAPE = {}'.format(Original_MAPE))
print('Ensemble_ML MAPE = {}'.format(Ensemble_MAPE_list[-1]))
print('--------')

