'''
Created on 13/02/2020

@author: asus
'''

import numpy as np, pandas as pd
import plotly.subplots as tls, plotly.figure_factory as plt
import plotly.graph_objs as go

from enum import Enum
import matplotlib

matplotlib.use('Agg')
import plotly.express as px
import seaborn as sns
import pathlib

path=pathlib.Path(__file__).parent.absolute()


''' ============================= '''
''' ====== A: LAYOUT UTILS ====== '''
''' ============================= '''

Button = Enum('Button', 'memory radio input checkbox time multidrop unidrop daterange graph figure html empty link text hours pie input_hidden  download dummy button')
colors = {'red':'#ed553b','yellow':'#f6d55c','dgreen':'#34988e','green':'#3caea3','blue':'#20639b'}
week_days = {'segunda':1,'terca':2,'quarta':3,'quinta':4,'sexta':5,'sabado':6,'domingo':7}
calendar = {'todos_dias':range(1,8),'dias_uteis':range(1,6),'fim_de_semana':range(6,8)}
hours=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
prioridade=['all','unknown',1,2,3,4,5,6,7,8,9]
meses=['Janeiro','Fevereiro','Março','Abril','Maio','Junho','Julho','Agosto','Setembro','Outubro','Novembro','Dezembro']
meses_dict={'Janeiro':1,'Fevereiro':2,'Março':3,'Abril':4,'Maio':5,'Junho':6,'Julho':7,'Agosto':8,'Setembro':9,'Outubro':10,'Novembro':11,'Dezembro':12}
typ=['Acidente Viação','Agressão','Alteração de Estado de Consciência','Criança/Recém-Nascido Doente','Diabetes','Dispneia','Dores no corpo','Hemorragia',
     'Intoxicação','Olhos/Ouvidos/Nariz/Garganta','Outros Problemas','Parto/Gravidez','Problemas Psiquiátricos/Suicídio','Queimadura / Electrocussão','Trauma']
granularidade=['mês','semana','dia da semana','dia']
ambiente=['humidade','direção vento', 'intensidade vento','precepitação acumulada','pressão atmosférica','radiação solar','temperatura']




''' =========================== '''
''' ====== B: PLOT UTILS ====== '''
''' =========================== '''


def get_series_plot(series,title,col=[],index=[]):
    #color=(sns.color_palette("Paired", len(series.columns))).as_hex()
    '''A: chart lines'''
    fig = tls.make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.00000001, horizontal_spacing=0.001)
    num=0
    if(len(index)==0):
        index=series[col].index

    if(len(col)==0):
        for col in series.keys():
            
            fig.append_trace({'x':index, 'y':series[col], 'type':'scatter', 'name':col }, 1, 1)
            num+=1
    else:
        fig.append_trace({'x':index, 'y':series[col], 'type':'scatter', 'name':col }, 1, 1)
            
    '''B: chart layout'''
    fig['layout'].update(dict(height=900,barmode='group',yaxis=dict(title=title),
                  xaxis=dict(title='tempo',autorange=True,rangeslider=dict(visible=True),tickangle=45,
                             rangeselector=dict(buttons=list([dict(step='all'),
                                     dict(stepmode='backward',step='hour',count=12,label='12 Horas',visible=True),
                                     dict(count=1,stepmode='backward',step='day',label='1 Dia',visible=True),
                                     dict(count=7,stepmode='backward',step='day',label='1 Week',visible=True),
                                     dict(count=1,stepmode='backward',step='month',label='1 Month',visible=True)])))))
    return fig

def get_series_plot_without_gaps(series,title,end,index,col=[]):
    #color=(sns.color_palette("Paired", len(series.columns))).as_hex()
    '''A: chart lines'''
    fig = tls.make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.00000001, horizontal_spacing=0.001)
    


    if(len(col)==0):
        mini=series['occurence.eid'].min()
        maxi=series['occurence.eid'].max()
        for col in series.columns:
            
            fig.append_trace({'x':index, 'y':series[col], 'type':'scatter', 'name':col }, 1, 1)
    else:
        mini=min(series[col])
        maxi=max(series[col])
        fig.append_trace({'x':index, 'y':series[col], 'type':'scatter', 'name':col }, 1, 1)
    

    lines=[]
    for line in end:
        lines.append({'type':'line','y0':mini-10,'y1':maxi,'x0':line,'x1':line,'line':dict(color="black",width=3,dash="dashdot")})
        
    '''B: chart layout'''
    fig['layout'].update(dict(shapes=lines,height=900,barmode='group',yaxis=dict(title=title),
                  xaxis=dict(type='category',title='tempo',autorange=True,rangeslider=dict(visible=True),tickangle=45,
                             rangeselector=dict(buttons=list([dict(step='all'),
                                     dict(stepmode='backward',step='hour',count=12,label='12 Horas',visible=True),
                                     dict(count=1,stepmode='backward',step='day',label='1 Dia',visible=True),
                                     dict(count=7,stepmode='backward',step='day',label='1 Week',visible=True),
                                     dict(count=1,stepmode='backward',step='month',label='1 Month',visible=True)])))))
    return fig


def add_predictor_series(fig, predictor):
    for var in predictor.variables:
        fig.append_trace(go.Scatter(name='Model['+var+']', x=predictor.index, yaxis='y1', y=predictor.series[var]["model"], mode='lines'),1,1)
        fig.append_trace(go.Scatter(name='Upper Bound['+var+']', x=predictor.index, yaxis='y1', y=predictor.series[var]["upperbound"], line=dict(color='rgb(68,68,68,0.2)', width=2, dash='dash')),1,1)
        fig.append_trace(go.Scatter(name='Lower Bound['+var+']', x=predictor.index, yaxis='y1', y=predictor.series[var]["lowerbound"], fill="tonexty", fillcolor='rgba(68,68,68,0.2)', line=dict(color='rgb(68,68,68,0.2)', width=2, dash='dash')),1,1)


def get_anomalies( predictor):
    for var in predictor.variables:
        n=len(predictor.series[var]["observed"])
        observed=predictor.series[var]["observed"]
        anomalies = np.array([np.NaN]*n)
        
        upperpositions = observed>predictor.series[var]["upperbound"][:n]

        anomalies[upperpositions] = observed[upperpositions]
    return anomalies
          
    
def add_anomalies(fig, variables,set,dates):
    colors=(sns.color_palette("Paired", len(variables))).as_hex()
    i=0
    shapes=[]
    for var in variables:
        if var!="time":
            fig.append_trace(go.Scatter(x=dates, y=set[var], mode='markers',marker=dict(symbol=i,color=colors[i]), name=var),1,1)
            i+=1
        else:
            for dat in [dates[i] if not np.isnan(set[var][i]) else 0 for i in range(len(set[var]))]:
                if dat!=0:
                    fig.add_shape(type='rect',xref='x',yref='paper',x0=dat,y0=0,x1=dat,y1=1,fillcolor='LightSalmon',opacity=1,layer='below',line=dict(color='LightSalmon',width=5))
    fig.update_traces(marker=dict(size=12,line=dict(width=2,color='DarkSlateGrey')),selector=dict(mode='markers'))
    fig.write_image("anomalies.pdf",width=1200)

def save(fig,name):
    fig.write_image(name+".pdf",width=1200)



