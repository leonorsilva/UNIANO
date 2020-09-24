'''
@info SARIMA predictor
@author Ines Leite and Rui Henriques
@version 1.0
'''

import pandas as pd
from itertools import product
from anomaly_detection import predictors
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import multiprocessing as mp
from joblib import Parallel
from joblib import delayed
from sklearn.metrics import mean_squared_error
import time
import threading
from sklearn.metrics import mean_absolute_error,median_absolute_error, mean_squared_error
import pathlib
from statsmodels.tsa.stattools import  acf, pacf,adfuller
from scipy import integrate

path=pathlib.Path(__file__).parent.absolute()

def f(x):
    return sum(x)

class Sarima(predictors.Predictor):

    def __init__(self, series, period,index, h=10, skip_cv=False,d=1, D=1 , Qs=1):
        super().__init__(series,index,h)
        self.d = d
        self.D = D
        self.Qs = Qs
        self.s = period
        self.skip_cv=skip_cv
        for var in self.variables:
            model = self.optimize_sarima(self.series[var]["observed"],var)
            model.plot_diagnostics()
            plt.savefig(str(path)+'diagnostico.png')
            
            result=model.get_prediction(start=0,end=len(self.series[var]["observed"]),full_results=True)
            

            predict=model.get_prediction(start=len(self.series[var]["observed"]),end=len(self.series[var]["observed"])+h, dynamic=True,full_results=True)
            
            forecast=predict.predicted_mean
            conf=predict.conf_int()
            
            res_conf=result.conf_int()[:,0]

            res_conf2=result.conf_int()[:,1]

            self.series[var]["model"]=np.append(result.predicted_mean,forecast)
            self.series[var]["lowerbound"]=np.append(res_conf,conf[:,0])
            self.series[var]["upperbound"]=np.append(res_conf2,conf[:,1])
            

    def optimize_sarima(self, series,var):
        """ d - integration order in ARIMA model
            D - seasonal integration order
            s - length of season """
        seriesoriginal=series
        series2=series
        Stationary=0
        SeasonalIntegrate=0
        
        regression='ct'
        integrate=0
        
        plt.figure()
        plt.plot(series, marker="o")
        plt.savefig(str(path)+'series.png')
        
        lag_acf = acf(series,nlags=self.s)
        lag_pacf = pacf(series, nlags=self.s)
        plt.figure()
        plt.plot(lag_acf, marker="o")
        plt.axhline(y=0,linestyle='--',color='gray')
        plt.axhline(y=-1.96/np.sqrt(len(series)),linestyle='--',color='gray')
        plt.axhline(y=1.96/np.sqrt(len(series)),linestyle='--',color='gray')
       
        plt.plot(lag_pacf, marker="o")
        plt.savefig(str(path)+'pacf1.png')
        
        
        if (lag_pacf[self.s]>1.96/np.sqrt(len(series)) or lag_pacf[self.s]<-1.96/np.sqrt(len(series))
                or lag_acf[self.s]>1.96/np.sqrt(len(series)) or lag_acf[self.s]<-1.96/np.sqrt(len(series))):
                                                                                                         
            print("pacf",lag_pacf[self.s-1],lag_pacf[self.s],lag_pacf[self.s])
            print("acf",lag_acf[self.s-1],lag_acf[self.s],lag_acf[self.s])
            SeasonalIntegrate=SeasonalIntegrate+1
            series=series[:-self.s]-series[self.s:]
            
        lag_acf = acf(series,nlags=self.s)
        print(lag_acf[1])
        if lag_acf[1]<0:  #demasiado diferenciada   antes -0.5
                print("demasiado diferenciada")
                SeasonalIntegrate=0
                series=series2
                
        while Stationary==0:
            result1 = adfuller(series, regression=regression, autolag='t-stat')#,maxlag=self.s)
            
            result = pd.Series(result1[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
             
            for key, value in result1[4].items():
                result['Critical Value (%s)' % key] = value
            print(result)
            if result['Test Statistic']<result['Critical Value (5%)']:
                Stationary=1
            elif integrate!=1 or SeasonalIntegrate==1:
                series=np.diff(series)
                integrate = integrate + 1
                #GroupedSalesDataDummy = GroupedSalesDataDummy.dropna(axis=0, how='any')
                regression='nc'
            else:
                diff=series[:-self.s]-series[self.s:]
                series=0
                series=diff
                SeasonalIntegrate=SeasonalIntegrate+1
                #GroupedSalesDataDummy = GroupedSalesDataDummy.dropna(axis=0, how='any')
            lag_acf = acf(series)
            if lag_acf[1]<-0.5:  #demasiado diferenciada
                print("demasiado diferenciada")
                integrate = integrate -1
                Stationary=1

        self.d=integrate
        self.D=SeasonalIntegrate
        ma=0
        ar=0
        index=0
        sar=0
        sma=0
        
       
        
        StartSeasonal=0
        for pacfvalue in lag_pacf[1:len(lag_pacf)]:
            index = index + 1
            if StartSeasonal==0:

                if pacfvalue>1.96/np.sqrt(len(series)) or pacfvalue<-1.96/np.sqrt(len(series)):
                    if ar<=10:
                        ar=ar+1
                    else:
                        StartSeasonal=1
                    if index==10 and ar==10:  #geometric decay
                        print("ar geometric decay")
                        ar=0
                        StartSeasonal=1
                    
                else:
                    StartSeasonal=1
            else:

                if index>=self.s and(pacfvalue>1.96/np.sqrt(len(series)) or pacfvalue<-1.96/np.sqrt(len(series))):
                    sar=1
                    break
               
        StartSeasonal=0
        index = 0
        for acfvalue in lag_acf[1:len(lag_acf)]:
            index = index + 1
            if StartSeasonal==0:

                if acfvalue>1.96/np.sqrt(len(series)) or acfvalue<-1.96/np.sqrt(len(series)):
                    ma=ma+1
                    if index==10 and ma==10:   #geometric decay
                        print("ma geometric decay")
                        ma=0
                        StartSeasonal=1
                else:
                    StartSeasonal=1
            else:

                if index>=self.s and(acfvalue>1.96/np.sqrt(len(series)) or acfvalue<-1.96/np.sqrt(len(series))):
                    sma=1
                    break
        
        param_values=[]

        if ar>2:
            ar=2
        if ma>2:
            ma=2
        if ar==0:
            ar=1
        if ma==0:
            ma=1
        for p in [0,ar]:
            for q in [0,ma]:
                for P in [0,1]: #[sar-1,sar,sar+1]:
                    for Q in [0,1]:#[sma-1,sma,sma+1]:
                        aux=[p,q,P,Q]
                        if p+q+P+Q==0:
                            continue
                        if P==1 and Q==1:
                            continue
                        if self.s==1 and p==0 and q==0:
                            continue
#                         if p!=0 and q!=0:
#                             continue
                        while aux[0]+aux[1]+aux[2]+aux[3]+self.d+self.D>6:
                            
                            i=aux.index(max(aux))
                            aux[i]=aux[i]-1
                        if aux not in param_values:
                            param_values.append(aux) 
                        

        
        results=[]
        series=[x for x in seriesoriginal]
        

        pool = mp.Pool(mp.cpu_count())
  

        for params in param_values:
            p=pool.apply_async(sarima, args=(self,series, params)) 
            results.append(p.get(timeout=10))
             
        pool.close()
        pool.join() 

        

        results.sort(key=lambda tup: tup[0])

        
        best_model,error=sarima(self,series,results[0][1],True,[mean_absolute_error])

        
        return best_model



def sarima(self, series, params, returnModel=False,loss=[]):

    print("try",params[0],params[1],params[2]) 
    try: #some combinations model fails to converge
        if(self.s>1):
            if(returnModel):
                errors=[]
                mean=[]
                aux=[]
                print(self.s)
                if not self.skip_cv:
                    for i in [2,1,0]:
                        train = series[0:len(series)-i*self.h]
                        print(len(train))
                        try:
                            model = sm.tsa.statespace.SARIMAX(train[0:len(train)-self.h], order=(params[0],self.d,params[1]), 
                                                      seasonal_order=(params[2],self.D,params[3],self.s)).fit(maxiter=20,disp=-1)
                            print(model.summary())
                            result=model.forecast(self.h)
                            for method in loss:
                                if method=="rmse":
                                    errors.append(self.root_mean_squared_error(result,train[-self.h:]))
                                else:
                                    errors.append(method(result,train[-self.h:]))
                        except Exception as error:
                            print("failure",error)
                            continue
                    if(len(loss)>1):
                        for index in range(0,len(loss)):
                            for i in range(index,len(errors),len(loss)):
                                
                                aux.append(errors[i])
                                print("index ",i,"errors ",aux)
                            mean.append(np.mean(aux))
                            mean.append(np.std(aux))
                            aux=[]
                        return model,mean
                    else:
                        return model,np.mean(np.array(errors))
                
                else:            
                    train = series
                    model = sm.tsa.statespace.SARIMAX(train, order=(params[0],self.d,params[1]), 
                                                  seasonal_order=(params[2],self.D,params[3],self.s)).fit(maxiter=20,disp=-1)
                    print(model.summary())
                    return model,[0]*6
                
            
            train = series
            model = sm.tsa.statespace.SARIMAX(train, order=(params[0],self.d,params[1]), 
                                              seasonal_order=(params[2],self.D,params[3],self.s)).fit(maxiter=20,disp=-1)
            
                
            print("succeed",params[0],params[1],params[2])
        else:
            if(returnModel):
                errors=[]
                mean=[]
                aux=[]
                if not self.skip_cv:
                    for i in [2,1,0]:
                        train = series[0:len(series)-i*self.h]
                        try:
                            model =ARIMA(train[0:len(train)-self.h], order=(params[0],self.d,params[1])).fit()
                            result=model.forecast(self.h)
                            for method in loss:
                                if method=="rmse":
                                    errors.append(self.root_mean_squared_error(result,train[-self.h:]))
                                else:
                                    errors.append(method(result,train[-self.h:]))
                        except Exception as error:
                            print("failure",error)
                            continue
                    if(len(loss)>1):
                        for index in range(0,len(loss)):
                            for i in range(index,len(errors),len(loss)):
                                
                                aux.append(errors[i])
                                print("index ",i,"errors ",aux)
                            mean.append(np.mean(aux))
                            mean.append(np.std(aux))
                            aux=[]
                        return model,mean
                    else:
                        train = series
                        model = ARIMA(train, order=(params[0],self.d,params[1])).fit()
                        print(model.summary())
                        return model,np.mean(np.array(errors))
                else:       
                    train = series
                    model = ARIMA(train, order=(params[0],1,params[1])).fit()
                    print(model.summary())
                    return model,[0]*6
                
            
            train = series
            model = ARIMA(train, order=(params[0],self.d,params[1])).fit()
            
                
            print("succeed",params[0],params[1],params[2])
            
    except Exception as error:
        print("failure",params[0],params[1],params[2],"=>",error)
        return (np.inf,params)
    return (model.aicc,params)



