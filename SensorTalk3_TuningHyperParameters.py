#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import warnings
import xgboost as xgb
from sklearn import  ensemble
from numpy import mean
from numpy import std
import time
from sklearn.linear_model import LinearRegression


# In[2]:


def cal_mape(y_ture, y_pred):
    return np.mean(np.abs((y_ture - y_pred) / y_ture)) * 100

def systematic_sampling(df, step):
 
    indexes = np.arange(0, len(df), step=step)
    systematic_sample = df.iloc[indexes]
    df = df.drop(indexes)
    return systematic_sample, df

def EnsembleMethod(STD_EC, STD_EC_Label, DUT_Validate, DUT_Validate_Label, DUT, DUT_Label,XGB_Best_n_estimators, XGB_Best_max_depth, XGB_Best_eta, XGB_Best_gamma, RF_Best_n_estimators, RF_Best_max_depth, RF_Best_min_samples_split, RF_Best_max_sampl):
    
    ## Start Time
    StartTime = time.time()
       
    XGB_Model = xgb.XGBRegressor(n_estimators = XGB_Best_n_estimators, 
                                 max_depth = XGB_Best_max_depth, 
                                 eta = XGB_Best_eta, 
                                 gamma = XGB_Best_gamma)
    
    XGB_Model.fit(STD_EC,STD_EC_Label)
    
    XGB_pred_train = XGB_Model.predict(DUT) ## 用測試集的結果
    XGB_pred_result = XGB_Model.predict(DUT_Validate) 
    
    #----------------------------------------------------------------------------------------------#
    RF_Model = ensemble.RandomForestRegressor(n_estimators = RF_Best_n_estimators, 
                                              max_depth = RF_Best_max_depth, 
                                              min_samples_split = RF_Best_min_samples_split,
                                              max_samples = RF_Best_max_sample,
                                              random_state = 1) ## cycle1 windowsize = 2
                                               
    
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
    Ensemble_mape = cal_mape(Ensemble_EC_Validate_label, Ensemble_result)
    
    EndTime = time.time()
    ExeTime = EndTime-StartTime
    
    return ExeTime, Ensemble_mape


# In[3]:


# evaluate a give model using cross-validation
def evaluate_model(model,ML_DUT2_Validate, ML_DUT2_Validate_Label):
    
    predict = model.predict(ML_DUT2_Validate)
    mape = cal_mape(ML_DUT2_Validate_Label, predict)
    return mape


# In[4]:


## Load Dataset
df_train = pd.read_excel("./InferenceData/Check_Moisture_Tempearutre/0404_DataFusion/0415_experiment_Window/Training_cycle1_W=2.xlsx")
#df_train = pd.read_excel("./Dataset1.xlsx")
df_train = df_train.reset_index(drop=True)

#--------------TrainData----------------------#
TrainingDataset = df_train.iloc[:]
TestingDataset,TrainingDataset = systematic_sampling(TrainingDataset,2)

#--------------Validation----------------------#
ValidationDataset = pd.read_excel("./InferenceData/Check_Moisture_Tempearutre/0404_DataFusion/0415_experiment_Window/Training_cycle2_W=2.xlsx")
#ValidationDataset = pd.read_excel("./Dataset3.xlsx")
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


# In[5]:


X = STD_EC
y = STD_EC_Label


# In[6]:


## XGBoost Hyper Parameters Function

def XGB_get_n_estimators():
    models = dict()
    for n in range(15,115,5):
        models[int(n)] = xgb.XGBRegressor(n_estimators=n)
    return models

def XGB_get_max_depth(Best_n_estimators):
    models = dict()
    for n in range(1,10):
        models[int(n)] = xgb.XGBRegressor(n_estimators=Best_n_estimators, max_depth = n)
    return models

def XGB_get_eta(Best_n_estimators,Best_max_depth):
    models = dict()
    for n in np.arange(0.01,0.301,0.01):
        models[float(n)] = xgb.XGBRegressor(n_estimators=Best_n_estimators, max_depth = Best_max_depth, eta = n)
    return models

def XGB_get_gamma(Best_n_estimators,Best_max_depth,Best_eta):
    models = dict()
    for n in np.arange(0,510,10):
        models[int(n)] = xgb.XGBRegressor(n_estimators=Best_n_estimators, max_depth = Best_max_depth ,eta = Best_eta,gamma = n)
    return models


# In[7]:


## Random Forest Hyper Parameters Function

def RF_get_n_estimators():
    models = dict()
    for n in np.arange(10, 140, 10):
        models[int(n)] = ensemble.RandomForestRegressor(n_estimators=n,random_state = 1)
    return models

def RF_get_max_depth(Best_n_estimators):
    models = dict()
    for n in np.arange(1, 22, 1):
        models[int(n)] = ensemble.RandomForestRegressor(n_estimators=Best_n_estimators, max_depth=n,random_state = 1)
    return models

def RF_get_min_samples_split(Best_n_estimators, Best_max_depth):
    models = dict()
    for n in np.arange(10, 510, 10):
        models[int(n)] = ensemble.RandomForestRegressor(n_estimators=Best_n_estimators, max_depth=Best_max_depth,  min_samples_split=n,random_state = 1)
    return models

def RF_get_max_samples(Best_n_estimators, Best_max_depth, Best_min_samples_split):
    models = dict()
    for n in np.arange(0.01, 0.26, 0.01):
        models[float(n)] = ensemble.RandomForestRegressor(n_estimators=Best_n_estimators, max_depth=Best_max_depth,  min_samples_split=Best_min_samples_split ,max_samples=n,random_state = 1)
    return models


# In[8]:


def Tuning_XGBOOST_HyperParameters():
    
    StartTime = time.time()
    
    # get_n_estimators
    models = XGB_get_n_estimators()
    # evaluate the models and store results
    n_estimators_parameters = dict()

    for name, model in models.items():
        model.fit(X,y)
        scores = evaluate_model(model, DUT_Validate, DUT_Validate_Label)
        n_estimators_parameters[name] = scores

    Best_n_estimators = min(n_estimators_parameters, key=n_estimators_parameters.get)
    
    # get_max_depth
    models = XGB_get_max_depth(Best_n_estimators)
    # evaluate the models and store results
    max_depth_parameters = dict()

    for name, model in models.items():
        model.fit(X,y)
        scores = evaluate_model(model, DUT_Validate, DUT_Validate_Label)
        max_depth_parameters[name] = scores

    Best_max_depth = min(max_depth_parameters, key=max_depth_parameters.get)
    
    # get_eta
    models = XGB_get_eta(Best_n_estimators,Best_max_depth)
    # evaluate the models and store results
    eta_parameters = dict()

    for name, model in models.items():
        model.fit(X,y)
        scores = evaluate_model(model, DUT_Validate, DUT_Validate_Label)
        eta_parameters[name] = scores

    Best_eta = min(eta_parameters, key=eta_parameters.get)   
    
    # get_gamma
    models = XGB_get_gamma(Best_n_estimators,Best_max_depth,Best_eta)
    # evaluate the models and store results
    gamma_parameters = dict()

    for name, model in models.items():
        model.fit(X,y)
        scores = evaluate_model(model, DUT_Validate, DUT_Validate_Label)
        gamma_parameters[name] = scores

    Best_gamma = min(gamma_parameters, key=gamma_parameters.get)
    
    EndTime = time.time()   
    ExeTime = EndTime-StartTime
    
    return Best_n_estimators, Best_max_depth, Best_eta, Best_gamma, ExeTime


# In[9]:


def Tuning_RandomForest_HyperParameters():
    
    StartTime = time.time()
    
    # get n_estimators
    models = RF_get_n_estimators()
    # evaluate the models and store results
    n_estimators_parameters = dict()
    for name, model in models.items():
        model.fit(X,y)
        scores = evaluate_model(model, DUT_Validate, DUT_Validate_Label)
        n_estimators_parameters[name] = scores

    Best_n_estimators = min(n_estimators_parameters, key=n_estimators_parameters.get)
    
    # get get_max_depth
    models = RF_get_max_depth(Best_n_estimators)
    # evaluate the models and store results
    max_depth_parameters = dict()

    for name, model in models.items():
        model.fit(X,y)
        scores = evaluate_model(model, DUT_Validate, DUT_Validate_Label)
        max_depth_parameters[name] = scores

    Best_max_depth = min(max_depth_parameters, key=max_depth_parameters.get)
    
    #get min_samples_split
    models = RF_get_min_samples_split(Best_n_estimators, Best_max_depth)
    # evaluate the models and store results
    min_samples_split_parameters = dict()

    for name, model in models.items():
        model.fit(X,y)
        scores = evaluate_model(model, DUT_Validate, DUT_Validate_Label)
        min_samples_split_parameters[name] = scores

    Best_min_samples_split = min(min_samples_split_parameters, key=min_samples_split_parameters.get)  
    
    # get max_samples
    models = RF_get_max_samples(Best_n_estimators, Best_max_depth, Best_min_samples_split)
    # evaluate the models and store results
    max_samples_parameters = dict()
    for name, model in models.items():
        model.fit(X,y)
        scores = evaluate_model(model, DUT_Validate, DUT_Validate_Label)
        max_samples_parameters[name] = scores

    Best_max_samples = min(max_samples_parameters, key=max_samples_parameters.get)

    EndTime = time.time()   
    ExeTime = EndTime-StartTime
    
    return Best_n_estimators, Best_max_depth, Best_min_samples_split, Best_max_samples, ExeTime


# In[10]:


XGB_Best_n_estimators, XGB_Best_max_depth, XGB_Best_eta, XGB_Best_gamma,XGB_ExeTime = Tuning_XGBOOST_HyperParameters()
print('XGBoost Best Hyper Parameters :\nn_estimators = {}\nmax_depth = {}\neta = {}\ngamma = {}'.format(XGB_Best_n_estimators, XGB_Best_max_depth, XGB_Best_eta, XGB_Best_gamma))

RF_Best_n_estimators, RF_Best_max_depth, RF_Best_min_samples_split, RF_Best_max_sample, RF_ExeTime = Tuning_RandomForest_HyperParameters()
print('Random Forest Best Hyper Parameters :\nn_estimators = {}\nmax_depth = {}\nmin_samples_split = {}\nmax_sample = {}'.format(RF_Best_n_estimators, RF_Best_max_depth, RF_Best_min_samples_split, RF_Best_max_sample))


# In[11]:


ValidationDataset = pd.read_excel("./InferenceData/Check_Moisture_Tempearutre/0404_DataFusion/0415_experiment_Window/Training_cycle8_W=2.xlsx")
#ValidationDataset = pd.read_excel("./Dataset3.xlsx")
ValidationDataset = ValidationDataset.iloc[:]
ValidationDataset = np.array(ValidationDataset, dtype=np.float64)

DUT_Validate = ValidationDataset[:,1:6] ## 驗證集 feature
DUT_Validate_Label = ValidationDataset[:,0] ## 驗證集 Label


# In[12]:


Ensemble_time,mape = EnsembleMethod(STD_EC, STD_EC_Label, DUT_Validate, DUT_Validate_Label, DUT, DUT_Label,XGB_Best_n_estimators, XGB_Best_max_depth, XGB_Best_eta, XGB_Best_gamma, RF_Best_n_estimators, RF_Best_max_depth, RF_Best_min_samples_split, RF_Best_max_sample)


# In[13]:


print('XGBoost ExeTime = {}(s)'.format(XGB_ExeTime))
print('Random Forest ExeTime = {}(s)'.format(RF_ExeTime))
print('Ensemble Training Time = {}(s)'.format(Ensemble_time))
print('Total time = {}(s)'.format(XGB_ExeTime+RF_ExeTime+Ensemble_time))
print('Performance = {}(%)'.format(mape))


# In[ ]:




