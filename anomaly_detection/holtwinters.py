# - coding: utf-8 --
'''
@info Holt-Winters predictor
@version 1.0
'''

import numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error,median_absolute_error, mean_squared_error
from anomaly_detection import predictors
from scipy.optimize import minimize
import math
#from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pip._internal.cli.cmdoptions import pre

''' ================================ '''
''' ====== A: HoltWinters ========== '''
''' ================================ '''

class HoltWinters(predictors.Predictor):
    
    ''' Holt-Winters model with the anomalies detection using Brutlag method
        - series: initial time series
        - period: length of a season
        - h, remove_h: predictions horizon and whether discounted or added to series
        - alpha, beta, gamma: HW model coefficients'''

    def __init__(self, series, period, index,h=10,  alpha=0.9, beta=0.9, gamma=0.9):
        super().__init__(series,index,h)
        self.scaling = 3
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.period = period

        for var in self.variables:
            self.get_optimized_holtwinters(var)


        
    def triple_exponential_smoothing(self, var,train=[]):
        
        '''A: init major variables'''
        deviation = 0
        if train==[]:
            
            timeseries = self.series[var]["observed"] 
        else: 
            timeseries =train
        
        result = [timeseries[0]]
        upperbound = [result[0]+self.scaling*deviation]
        lowerbound = [result[0]-self.scaling*deviation]
        
        trend = 0.0
        for i in range(self.period): 
            trend += float(timeseries[i+self.period]-timeseries[i])/float(self.period)
        trend = trend/float(self.period)
        smooth=float(timeseries[i+self.period])/float(self.period)

        '''B: init seasonal components'''
        seasonals = {}
        n_seasons = int(len(timeseries)/self.period)
        
        season_averages = []
        for j in range(n_seasons): #season averages
            
            mysum = sum(timeseries[self.period*j : self.period*j+self.period])
            season_averages.append(mysum/float(self.period))
        for i in range(self.period):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += timeseries[self.period * j + i] - season_averages[j]

            seasonals[i] = sum_of_vals_over_avg / n_seasons

        '''C: fitting model'''
        for i in range(0,len(timeseries)+self.h):

            if i >= len(timeseries):
                
                '''C1: prediction'''
                m = i - len(timeseries) + 1
                result.append((smooth+m*trend)+seasonals[i%self.period])
                deviation = deviation*1.01 #uncertainty increases along pred horizon

            else:
                
                '''C2: modeling'''
                val = timeseries[i]
                last_smooth, last_trend = smooth, trend
                smooth = self.alpha*(val-seasonals[i%self.period])+(1-self.alpha)*(last_smooth + last_trend)
                trend = self.beta*(smooth-last_smooth)+(1-self.beta)*last_trend
                seasonals[i % self.period] = self.gamma * (val - smooth) + (1 - self.gamma) * seasonals[i % self.period]
                result.append(smooth + trend + seasonals[i % self.period])
                # Deviation is calculated according to Brutlag algorithm.
                deviation = 0.1 * np.abs(timeseries[i] - result[i])+ (1 - 0.1) * deviation
                #self.season.append(seasonals[i % self.period])

            '''C3: compute bounds'''
            upperbound.append(result[-1]+self.scaling*deviation)
            lowerbound.append(result[-1]-self.scaling*deviation)
        
        '''D: save model'''

        self.series[var]["model"]=result
        
        self.series[var]["lowerbound"]=lowerbound
        self.series[var]["upperbound"]=upperbound


    
    def print_statistics(self):
        print("alpha=",self.alpha," beta=",self.beta," gamma=",self.gamma)
        for var in self.variables:
            print("====%s===="%var)
            print("Observed   :",self.series[var]["observed"])
            print("Modeled    :",self.series[var]["model"])
            print("Lower bound:",self.series[var]["lowerbound"])
            print("Upper bound:",self.series[var]["upperbound"])

    
    
    def get_optimized_holtwinters(self,variable):

        '''A: Minimizing loss function'''
       
        opt = minimize(series_cv_score, x0=[0.1,0.1,0.1], args=(self,[mean_absolute_error]),method="TNC",bounds=((0,0.6),(0,0.6),(0,0.6)))
        
        alpha_final, beta_final, gamma_final = opt.x
        self.alpha = alpha_final
        self.beta = beta_final
        self.gamma = gamma_final
        metrics=["rmse",mean_absolute_error,median_absolute_error]
        
        error=series_cv_score([alpha_final, beta_final, gamma_final],self,metrics)
        index=0
        for var in range(0,len(self.variables)):
            result = "===== %s ====\nTesting\n RMS: %f +/- %f\nMAE: %f +/- %f\nMDAE: %f +/- %f\n" % (self.variables[0],error[index+var],error[index+var+1],error[index+var+2],error[index+var+3],error[index+var+4],error[index+var+5])
        index+=5
        result+="Convergence: %s \n alpha %f\n beta %f\n gamma %f\n" %(opt.success,alpha_final,beta_final,gamma_final)
        self.test=result
        self.triple_exponential_smoothing(variable)
        
        
        
        #return HoltWinters(series, period=period, h=h, remove_h=True, alpha=alpha_final, beta=beta_final, gamma=gamma_final)

''' ==================================== '''
''' ====== B: HW Optimization ========== '''
''' ==================================== '''

def series_cv_score(params, predictor, loss):
    errors = []
    predictor.alpha = params[0]
    predictor.beta = params[1]
    predictor.gamma = params[2]
    mean=[]
    aux=[]
    for var in predictor.variables:    
        series=predictor.series[var]["observed"]
        

        for i in [2,1,0]:
            train = series[0:len(series)-i*predictor.h]

            try:
                predictor.triple_exponential_smoothing(var,train[0:len(train)-predictor.h])
            except Exception as error:
                continue
            for method in loss:
            
                if method=="rmse":
                    errors.append(predictor.root_mean_squared_error(predictor.series[var]["model"][-predictor.h:],train[-predictor.h:]))
                else:

                    errors.append(method(predictor.series[var]["model"][-predictor.h:],train[-predictor.h:]))

    if(len(loss)>1):
        for index in range(0,len(loss)):
            for i in range(index,len(errors),len(loss)):
                aux.append(errors[i])
            mean.append(np.mean(aux))
            mean.append(np.std(aux))
            aux=[]
    else:
        return np.mean(np.array(errors))
    return mean#np.mean(np.array(errors))


