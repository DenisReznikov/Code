import numpy as np
import pandas as pd


def prepare_data(data,period,label):

    Test = data[-period : ]
    Train = data[: -period] 
    train_x = Train.drop([label], axis=1)
    train_y = Train[label].to_numpy()
    test_x = Test.drop([label], axis=1)
    test_y = Test[label].to_numpy()  

    return train_x, test_x, train_y, test_y

