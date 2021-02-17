
import numpy as np
import pandas as pd
import logging 
import timeit
from catboost import CatBoostRegressor
from utils import prepare_data

class CatBoost_model:

    def __init__(self):
        self.__cat = CatBoostRegressor()
        logging.getLogger('CatBoost').setLevel(logging.ERROR)
        logging.basicConfig(filename='xgboost.log',
                            format='%(levelname)s %(asctime)s: %(message)s',
                            level=logging.DEBUG)
        
    
    
    def predict(self, data, period_for_predict = 1, label = "Status"):

        self.__period = period_for_predict
        self.__label = label

        if not isinstance(data, pd.DataFrame):
            logging.error("Input should be DataFrame")
            raise TypeError
        
        if self.__period < 1:
            logging.error("Period for predict less then 1")
            raise ValueError

        start = timeit.default_timer()
        
        X_train, X_test, y_train, y_test = prepare_data(data,self.__period,self.__label)

        self.__cat.fit(X_train, y_train, verbose_eval=False)
        prediction =  self.__cat.predict(X_test)

        stop = timeit.default_timer()
        timing = np.round(stop - start, 3)
        logging.info(f'Time for CatBoost "{timing}"')
        return prediction
    
    
    def __print_tree(self,model):
        pass