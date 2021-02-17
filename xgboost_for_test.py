import numpy as np
from xgboost import XGBRegressor
import pandas as pd
import logging 
import timeit
from utils import prepare_data

class XGB_model:

  def __init__(self):
    self.__xgb = XGBRegressor()


  def prepare_data(Train,Test,period,label):
    train_x = Train.drop([label], axis=1)
    train_y = Train[label].to_numpy()
    test_x = Test.drop([label], axis=1)
    test_y = Test[label].to_numpy()  

    return train_x, test_x, train_y, test_y

    

  def predict(self, Train, Test, period_for_predict = 1,label = "Status"):
    self.__period = period_for_predict
    self.__label = label
    
    start = timeit.default_timer()
    
    X_train, X_test, y_train, y_test = prepare_data(data,self.__period,self.__label)

    self.__xgb.fit(X_train, y_train)
    prediction =  self.__xgb.predict(X_test)
	
    return prediction

