'''
@info abstract class for predictors
@version 1.0
'''

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error
import pandas as  pd


class Predictor(object):

    def __init__(self,series,index, h):
        self.h = h
        self.series = {}
        self.variables = series.columns
        self.index = index
        self.test=''
        for var in self.variables:
            ts = series[var].values
            self.series[var]={"observed":ts,"model":None,"lowerbound":None,"upperbound":None}

            
    def __end__(self):
        self.series = None

    def get_errors(self):
        result = ''
        for var in self.variables:
            ts_observed = self.series[var]["observed"]
            
            ts_modeled = self.series[var]["model"][0:len(ts_observed)]
            
            rmse = self.root_mean_squared_error(ts_modeled,ts_observed)
            mape = 0 #self.mean_absolute_percentage_error(ts_observed,ts_modeled)
            mae = mean_absolute_error(ts_modeled,ts_observed)
            mdae=median_absolute_error(ts_modeled,ts_observed)
            result += "===== %s ====\n Training\nRMS: %f\nMAPE: %f\nMAE: %f\nMDAE: %f\n" % (var,rmse,mape,mae,mdae)
            result+=self.test
        return result

    def mean_absolute_percentage_error(self,y_pred,y_true): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def root_mean_squared_error(self,y_pred,t_true):
        return np.sqrt(mean_squared_error(y_pred,t_true))
