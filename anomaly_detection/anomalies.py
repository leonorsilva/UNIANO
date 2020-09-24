
#%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from copy import deepcopy
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from anomaly_detection import holtwinters,sarima
from anomaly_detection import LSTM
from scipy.spatial import distance
import pathlib
from numpy.f2py.rules import aux_rules

from anomaly_detection import gui_utils as gui
from scipy.stats import zscore,pearsonr
import math
import statsmodels.robust
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import kstest,gamma,exponweib,weibull_min,lognorm,pareto,norm,halfnorm,boxcox
from statsmodels.tsa.seasonal import seasonal_decompose

path=pathlib.Path(__file__).parent.absolute()

def threshold(data,name,thres,maxi=None):

    data=[d for d in data if not(d is None or np.isnan(d))]

    
    count, bins, ignored = plt.hist(data, align='mid')
    count.sort()

    mini=max(data)
    
    dist=[]
    threshold=0.95
    
    if thres=='Gamma distribution':
        if maxi is None:
            
            d=np.linspace(0,max(data)+1,1000)
        else:
            d=np.linspace(0,maxi+1,1000)
        
        loc,scale=norm.fit(data)
        v2,p=kstest(data,'norm',norm.fit(data))
        
        
        a2,loc2,scale2=gamma.fit(data)
        v2,p2=kstest(data,'gamma',gamma.fit(data))



        
        d.sort()

        if name=='time':
            threshold=0.95
            func=norm.pdf(d,loc=loc,scale=scale)
            values=norm.cdf(d,loc=loc,scale=scale)
            dist=values
        else:
            func=gamma.pdf(d,a2,loc=loc2,scale=scale2)
            values=gamma.cdf(d,a2,loc=loc2,scale=scale2)
            dist=values
            
    
        for i in range(0,len(values)):
            if values[i]>=threshold :
                mini=d[i]
                break
        
    elif thres=='Standard deviation':
        median=np.mean(data)
        mad=np.std(data)
        mini=median+2*mad
        
    
    else:
        ecdf = ECDF(data)
        d=data.copy()
        d.sort()

        for index in range(0,len(ecdf.y)):
            if ecdf.y[index]>=0.98 :
                mini=ecdf.x[index]
                break

    return mini,dist


  
'''calculate mahalanobis distance'''
def Mahala_distantce(x,mean,cov):
    d = np.dot(x-mean,np.linalg.inv(cov))
    d = np.dot(d, (x-mean).T)
    return d
  

      
def slidingwindow(data,window_size,period):

    count=int(np.isnan(data).sum()/2)
    if count>0:
        data=data[~np.isnan(data)]
        data = data.reshape(len(data), -1)

    m_dist=[0]*(count+1)
    
    median=np.median(data)
    covtotal = 0

    
    meantotal = sum(data)/len(data)

    covtotal = 0
    for e in data:
        
        covtotal += np.dot((e-meantotal).reshape(1, 1), (e-meantotal).reshape(1, 1))

    covtotal /= len(data)
    
    
    '''random walk calculation'''
    covdiff = 0
    meandiff=0
    for i in range(1,len(data)):
        
        meandiff += data[i]-data[i-1]

    meandiff=meandiff/(len(data)-1)
    
    for i in range(1,len(data)):   
        covdiff += np.dot(((data[i]-data[i-1])-meandiff).reshape(1, 1), ((data[i]-data[i-1])-meandiff).reshape(1, 1))
    covdiff /= (len(data)-1)
    
    for i in range(1,len(data)):
        p=data[i]
        p1=data[i-1]
        

        m_dist.append(Mahala_distantce(p-p1,meandiff,covdiff))
    for i in range(count):
        m_dist.append(0)

    mean_dist=[0]*(3)
    
    
    '''clusters calculation'''
    cluster2=[0]*(8+count)
    cluster3=[0]*(8+count)
    x=[]
    y=[]
    for i in range(window_size,len(data)-window_size):
        if i==len(data)-window_size:
             
            aux=data[i-window_size:]
        else:
            aux=data[i-window_size:i+window_size]
        
        
        mean2=(sum(aux)-p)/(len(aux)-1)
        x.append(data[i])
        y.append(mean2)
        
    cov = 0
    for index in range(0,len(x)):
        
        cov += np.dot((x[index]-y[index]).reshape(1, 1), (x[index]-y[index]).reshape(1, 1))
    cov /= len(aux)
    
    for i in range(8,len(data)-8):    
        
        p=data[i]
        p_1=data[i-1]
        p_2=data[i-2]
        p_3=data[i-3]
        p_4=data[i-4]
        p_5=data[i-5]
        p_6=data[i-6]
        p_7=data[i-7]
        p_8=data[i-8]

        p1=data[i+1]
        p2=data[i+2]
        p3=data[i+3]
        p4=data[i+4]
        p5=data[i+5]
        p6=data[i+6]
        p7=data[i+7]
        p8=data[i+8]

        
        mean_dist.append(Mahala_distantce(p,mean2,cov))
        euc_dist=[abs(p-p_4),abs(p-p_3),abs(p-p_2),abs(p-p_1),abs(p-p1),abs(p-p2),abs(p-p3),abs(p-p4)]
        euc_dist_1=[abs(p_1-p_5),abs(p_1-p_4),abs(p_1-p_3),abs(p_1-p_2),abs(p_1-p1),abs(p_1-p2),abs(p_1-p3)]
        euc_dist_2=[abs(p_2-p_6),abs(p_2-p_5),abs(p_2-p_4),abs(p_2-p_3),abs(p_2-p_1),abs(p_2-p1),abs(p_2-p2)]
        euc_dist_3=[abs(p_3-p_7),abs(p_3-p_6),abs(p_3-p_5),abs(p_3-p_4),abs(p_3-p_2),abs(p_3-p_1),abs(p_3-p1)]
        euc_dist_4=[abs(p_4-p_8),abs(p_4-p_7),abs(p_4-p_6),abs(p_4-p_5),abs(p_4-p_3),abs(p_4-p_2),abs(p_4-p_1)]
        euc_dist1=[abs(p1-p_3),abs(p1-p_2),abs(p1-p_1),abs(p1-p2),abs(p1-p3),abs(p1-p4),abs(p1-p5)]
        euc_dist2=[abs(p2-p_2),abs(p2-p_1),abs(p2-p1),abs(p2-p3),abs(p2-p4),abs(p2-p5),abs(p2-p6)]
        euc_dist3=[abs(p3-p_1),abs(p3-p1),abs(p3-p2),abs(p3-p4),abs(p3-p5),abs(p3-p6),abs(p3-p7)]
        euc_dist4=[abs(p4-p1),abs(p4-p2),abs(p4-p3),abs(p4-p5),abs(p4-p6),abs(p4-p7),abs(p4-p8)]
        aux=[euc_dist_4,euc_dist_3,euc_dist_2,euc_dist_1,euc_dist1,euc_dist2,euc_dist3,euc_dist4]
        
        
        aux2=[]
        for vector in aux:
            vector.sort()
        for vector in aux:
            aux2.append(aux)
        
        values=[]
        values3=[]
        valuesouter=[]
        valuesouter3=[]
        
        cluster2.append(np.mean(euc_dist)/np.mean(aux2))
        low=[]
        euc_dist2=euc_dist.copy()
        aux2=aux.copy()
        
        for j in range(0,8):
            values3.append(min(euc_dist))
            valuesouter3.append(np.mean(aux[euc_dist.index(values3[j])]))
            
            
            aux.remove(aux[euc_dist.index(values3[j])])
            euc_dist.remove(values3[j])
            res=(values3[j])/valuesouter3[j]
            

            low.append(res)

        low=[0 if math.isnan(low[i]) else low[i] for i in range(len(low))]

        low.sort()
        
        cluster3.append(np.mean(low))

    '''variance from mean and median'''
    mean_dist=np.concatenate((mean_dist,[0]*(1+count)))
    cluster2=np.concatenate((cluster2,[0]*(8+count)))
    cluster3=np.concatenate((cluster3,[0]*(8+count)))
    

    mean_dist2=[0]*(len(data)+2*count)
    median_dist2=[0]*(len(data)+2*count)
    x=[]

    mean = sum(data)/len(data)
    median=np.median(data)


    
    mad=np.median([abs(data[i]-median) for i in range(0,len(data))])

    cov=0
    covmedian=0
    for index in range(0,len(data)):
            
        cov += np.dot((data[index]-mean).reshape(1, 1), (data[index]-mean).reshape(1, 1))
        covmedian += np.dot((data[index]-median).reshape(1, 1), (data[index]-median).reshape(1, 1))
    cov /= len(data)
    covmedian /= len(data)
        
     
    for i in range(0,len(data)):
        if mean==0:
            mean_dist2[i]=0
        elif mad==0:
            median_dist2[i]=0
        else:
            p=data[i]
                
            mean_dist2[i+count]=Mahala_distantce(p,mean,cov)
            median_dist2[i+count]=Mahala_distantce(p,median,covmedian)

    return m_dist,mean_dist2,median_dist2,cluster2,cluster3
    
    
def update_chart(testset,seriesmeteo,period,minutes,timer,gaps=False,end=[]):

    df=testset

    step=math.ceil((len(df))/20)

    lstm=LSTM.lstm()
    x_train_size=10
    lstm.fit(period,minutes,['number_occurences'],save_dir=str(path),window_size=7*step, moving_step=step, epochs=200,
                  loss_function="mae", batch_size=2, series_df=df,isntance_moving_step=1, x_train_size=x_train_size, y_train_size=1,l_r=0.01,reg_type=1,reg_value=0.1)
    

    res=lstm.anomalies()

    res=res[:-1]
    errors = [abs(x-y) for x,y in zip(df.iloc[x_train_size:,0],res)]

    mean = sum(errors)/len(errors)

    cov = 0
    for e in errors:
        
        cov += np.dot((e-mean).reshape(len(e), 1), (e-mean).reshape(1, len(e)))
    cov /= len(errors)
    lstmerror=[0]*x_train_size
    for e in errors:
        lstmerror.append(Mahala_distantce(e,mean,cov))
    lstm=[0]*x_train_size
    for x in res:
        lstm.append(x[0])

    
    df=testset
    df=df.rename(columns={'number_occurences':'holtwinters'})
    h=round(len(testset.index)/4)

    predictor = holtwinters.HoltWinters(series=df,period=period,index=df.index,h=h)
    df=df.rename(columns={'holtwinters':'sarima'})
    predictor2 = sarima.Sarima(series=df,period=period,index=testset.index,h=h,skip_cv=False)
    

    if period==1:
        result = seasonal_decompose(testset, model='additive')
    else:
        result = seasonal_decompose(testset['number_occurences'], model='additive',period=period)
    testset['residuo']=result.resid

    testset2 = testset['residuo'].values.reshape(len(testset['residuo']), -1)

    
    
    period=1    
    slide,mean2,median2,cluster2,cluster3=slidingwindow(testset2,2,period)
    lstmerror=[np.NaN if np.isnan(testset['residuo'][i]) or testset['residuo'][i]<0 else lstmerror[i] for i in range(len(lstmerror))]
    slide=[np.NaN if np.isnan(testset['residuo'][i]) or testset['residuo'][i]<0 else slide[i] for i in range(len(slide))]
    mean2=[np.NaN if np.isnan(testset['residuo'][i]) or testset['residuo'][i]<0 else mean2[i] for i in range(len(mean2))]
    median2=[np.NaN if np.isnan(testset['residuo'][i]) or testset['residuo'][i]<0 else median2[i] for i in range(len(median2))]
    cluster2=[np.NaN if np.isnan(testset['residuo'][i]) or testset['residuo'][i]<0 else cluster2[i] for i in range(len(cluster2))]
    cluster3=[np.NaN if np.isnan(testset['residuo'][i]) or testset['residuo'][i]<0 else cluster3[i] for i in range(len(cluster3))]

    testset['lstm'] =lstm



    
    if gaps:
        
        fig=gui.get_series_plot_without_gaps(testset,"serie",end,testset.index)
    else:
        fig=gui.get_series_plot(testset,"serie")
    testset=testset.join(timer)
    testset['holtwinters']=gui.get_anomalies(predictor)
    testset['sarima']=gui.get_anomalies(predictor2)      
    


    distribution={'lstm':lstmerror,'random_walk':slide,'mean_period':mean2,'median_period':median2,'cluster_local':cluster2,'cluster_global':cluster3}
    
    distribution=pd.DataFrame(distribution)
    

    
    data={'index':testset['number_occurences'].index,'end':end,'gaps':gaps,'testset':dict(testset),'distribution':dict(distribution),'tempo':dict(seriesmeteo)}
    
    thres='Gamma distribution'
    maximum=distribution.values.max()

    serie=pd.DataFrame()
    _,serie['lstm_distribtuion']=threshold(distribution['lstm'],'lstmerror.png',thres,maximum)

    _,serie['random_walk_distribtuion']=threshold(distribution['random_walk'],'random_walk.png',thres,maximum)
    _,serie['mean_distribution']=threshold(distribution['mean_period'],'slide_mean_period.png',thres,maximum)
    _,serie['median_distribution']=threshold(distribution['median_period'],'slide_median_period.png',thres,maximum)
    _,serie['cluster2_distribution']=threshold(distribution['cluster_local'],'cluster2.png',thres,maximum)
    _,serie['cluster3_distribution']=threshold(distribution['cluster_global'],'cluster3.png',thres,maximum)
    _,serie['time_distribution']=threshold(testset['time'],'time.png',thres,maximum)
    serie.index=np.linspace(0,maximum+1,1000) 

    data['serie']=serie


    gui.add_predictor_series(fig,predictor)   
    gui.add_predictor_series(fig,predictor2)
    gui.save(fig,'series')
    

    anomalies(data,thres)
    
    


    
def anomalies(data,thres):
    
    testset=data['testset']
    distribution=data['distribution']
    index=data['index']

    fig=gui.get_series_plot(testset,"serie",col='number_occurences',index=index)
    

    min,_ =threshold(distribution['lstm'],'lstmerror.png',thres)   
    testset['lstm']= pd.DataFrame([ np.NaN if distribution['lstm'][x] is None or np.isnan(distribution['lstm'][x]) or np.isnan(distribution['lstm'][x]) or distribution['lstm'][x]<min else testset['number_occurences'][x] for x in range(0,len(distribution['lstm']))])[0] 
    
    
    min,_=threshold(distribution['random_walk'],'random_walk.png',thres)
    testset['random_walk'] =pd.DataFrame([np.NaN if distribution['random_walk'][x] is None or np.isnan(distribution['random_walk'][x]) or distribution['random_walk'][x]<min else testset['number_occurences'][x] for x in range(0,len(distribution['random_walk']))])[0] 
    
    min,_=threshold(distribution['mean_period'],'slide_mean_period.png',thres) 
    testset['mean_period'] = pd.DataFrame([ np.NaN if distribution['mean_period'][x] is None or np.isnan(distribution['mean_period'][x]) or distribution['mean_period'][x]<min else testset['number_occurences'][x] for x in range(0,len(distribution['mean_period']))])[0] 

    min,_=threshold(distribution['median_period'],'slide_median_period.png',thres)
    #print('slide_median_period',distribution['slide_median_period'])
    testset['median_period'] = pd.DataFrame([np.NaN if  distribution['median_period'][x] is None or np.isnan(distribution['median_period'][x]) or distribution['median_period'][x]<min else testset['number_occurences'][x] for x in range(0,len(distribution['median_period']))])[0] 

    min,_=threshold(distribution['cluster_local'],'cluster2.png',thres)
   
    testset['cluster_local'] =pd.DataFrame([ np.NaN if  distribution['cluster_local'][x] is None or np.isnan(distribution['cluster_local'][x]) or distribution['cluster_local'][x]<min else testset['number_occurences'][x] for x in range(0,len(distribution['cluster_local']))])[0] 
    
    min,_=threshold(distribution['cluster_global'],'cluster3.png',thres)
    testset['cluster_global'] =pd.DataFrame( [np.NaN if  distribution['cluster_global'][x] is None or np.isnan(distribution['cluster_global'][x]) or distribution['cluster_global'][x]<min else testset['number_occurences'][x] for x in range(0,len(distribution['cluster_global']))])[0] 

    scaler = MinMaxScaler()
    timeaux=scaler.fit_transform(np.array(testset['time']).reshape(-1,1))

    
    min,_=threshold(testset['time'],'time',thres)
    testset['time'] =pd.DataFrame([ np.NaN if testset['time'][x] is None or testset['time'][x]<min else testset['number_occurences'][x] for x in range(0,len(testset['time']))])[0] 

    ecdf=pd.DataFrame.from_dict(distribution ,orient='columns')
    ecdf['holt']=[0 if val is None else 1 for val in testset['holtwinters']]
    ecdf['sarima']=[0 if val is None else 1 for val in testset['sarima']]

    anomaly2=[]
    ii=0
    for _,row in ecdf.iterrows():
        i=ecdf.ge(row)
        
        if (len(ecdf[i].dropna())==0):
            anomaly2.append(np.NaN)
        else:
            anomaly2.append(len(ecdf[i].dropna())/len(ecdf))

        ii+=1
    

    testset['anomaly_0.015']=[testset['number_occurences'][i] if anomaly2[i]<=0.015 else np.NaN for i in range(len(anomaly2))]
    testset['anomaly_0.05']=[testset['number_occurences'][i] if anomaly2[i]<=0.05 else np.NaN for i in range(len(anomaly2))]
    
    columns=['lstm','holtwinters','sarima','random_walk','mean_period','median_period','cluster_local','cluster_global','time','anomaly_0.015','anomaly_0.05']
    
    mask=[0]*len(testset['cluster_global'])
 
    for col in columns:
        for idx in range(0,len(mask)):
            if (not testset[col][idx] is None or np.isnan(testset[col][idx])) and testset[col][idx]>0:
                mask[idx]=1

  
    
    
    '''desalinhar bolinhas anomalias'''
    maxi=max(testset['number_occurences'])
    
    offset=0
    
    aux=pd.DataFrame()
    mask_idx=[]
    for col in columns:
        aux2=[]
        for idx in range(0,len(mask)):
            if mask[idx]==1:
                mask_idx.append(idx)
                aux2.append(testset[col][idx])
        aux[col]=aux2
        aux[col]=[0 if (aux[col][row] is None or math.isnan(aux[col][row]) or aux[col][row]==0) else 1 for row in range(0,len(aux[col]))]

        matrix=[]



    columnss=['lstm','holtwinters','sarima','random_walk','mean_period','median_period','cluster_local','cluster_global','time']

    '''Kappa calculation'''    

    for i in range(0,len(columnss)):
        matrix.append([0]*len(columnss))
    for col in range(0,len(columnss)):
        for row in range(0,len(columnss)):
            num1=[1 if aux[columnss[row]][i]==aux[columnss[col]][i] and aux[columnss[col]][i]==1 else 0 for i in range(len(aux[columnss[col]]))]
            num0=[1 if aux[columnss[row]][i]!=aux[columnss[col]][i] else 0 for i in range(len(aux[columnss[col]]))]
            matrix[col][row]=(np.sum(num1)/(np.sum(num1)+np.sum(num0)))



    kappa=[]
    anom=[]
    weight=[]
    columnss.remove('time')
    aux=aux.drop('time',axis=1)
    aux=aux.drop('anomaly_0.015',axis=1)
    aux=aux.drop('anomaly_0.05',axis=1)
    for _,row in aux.iterrows():
        
        prob_obs=(sum(row)/len(row))
        
        kappa.append(prob_obs)
        indexx=np.where(row!=0)[0]
        
        mini=1
        for r in indexx:
            for c in indexx:
                if c==r:
                    continue
                if matrix[c][r]<mini:
                    mini=matrix[c][r]
        weight.append(mini)

    for col in columnss:
        anom.append((sum(aux[col])/(len(row)*len(aux)))**2)


    prob_espe=sum(anom)
    kappa=[(kappa[i]-weight[i]*prob_espe)/(1-weight[i]*prob_espe) for i in range(len(kappa))]

    idx=[i for i, val in enumerate(kappa) if val >= 0.4]   
    

    
    testset['fleiss']=[None]*len(testset['cluster_local'])
    for i in idx:
        testset['fleiss'][mask_idx[i]]=testset['number_occurences'][mask_idx[i]]  
    copy=testset.copy()
    columns.append('fleiss')


    '''calculation of height for each marker on the plot'''
    for idx in range(0,len(testset['number_occurences'])):
        for col in columns:

            if not (copy[col][idx] is None or math.isnan(copy[col][idx])):
                if idx>0:

                    
                    if testset['number_occurences'][idx-1]<testset['number_occurences'][idx]:
                        testset[col][idx]=copy[col][idx]+offset
                    else:
                        testset[col][idx]=copy[col][idx]-offset
                else:
                    if testset['number_occurences'][idx+1]<testset['number_occurences'][idx]:
                        testset[col][idx]=copy[col][idx]+offset
                    else:
                        testset[col][idx]=copy[col][idx]-offset
                offset=offset+maxi/20

        offset=0
    
    
    gui.add_anomalies(fig,['lstm','holtwinters','sarima','random_walk','mean_period','median_period','cluster_local','cluster_global','fleiss','time','anomaly_0.015','anomaly_0.05'],testset,index)
    

if __name__ == '__main__':


    series = pd.read_csv("series.csv",index_col=0,parse_dates=True) 
    timer=pd.read_csv("timer.csv",index_col=0,parse_dates=True)
    
    period=1
    minutes=1440
    

    update_chart(series,[],period,minutes,timer)
     
