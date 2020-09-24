import tensorflow as tf

from keras.losses import mean_squared_error
from keras.losses import mean_absolute_error
from keras.losses import mean_absolute_percentage_error
from keras.losses import  mean_squared_logarithmic_error
#from keras.losses import cosine_similarity


import numpy as np
import math
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd


import statistics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import os
import shutil
import json
import csv
#from data2help.gui import utilplot as util




SAVE_MODEL_PATH = "/home/leonor/"
DAY_MINUTE = 1440
DATEFORMAT = "%Y-%m-%d %H:%M:%S"

DATASET_FILE_NAME = "dataset_"
HISTORY_FILE_NAME = "history_dataset_"
SCORE_FILE_NAME = "score_dataset_"


DICT_LOSS_FUNCTION = { "mae":mean_absolute_error,
                      "mse":mean_squared_error,
                      "mape":mean_absolute_percentage_error,
                      "msle":mean_squared_logarithmic_error}


class lstm():

    def __init__(self):
        self.series_df = None
        self.batch_size = None
        self.output_variables = None
        self.series = None
        self.period = None
        self.interval_minute = None
        '''Training data'''
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        '''Data settings'''
        self.window_size = 27
        self.moving_step = 1
        self.x_train_size = 2
        self.y_train_size = 1
        self.test_perc = 0.9
        self.x_val = None
        self.y_val = None
        '''Error array'''
        self.score_mae = []
        self.score_mse = []
        self.score_rmse = []
        self.score_msle=[]
        '''History'''
        self.history = []
        '''Linechart'''
        self.y_test_y_hat = []
        self.scale=[]
    '''
        Period = NUmber of data point in a day
        interval_minute = time between datapoint
        moving_steo = desfasamento entre datasets
       isntance_moving_step =  desfasamento entre instancias 
    '''
    def fit(self, period, interval_minute, output_variables, window_size, moving_step, save_dir,
                            batch_size =2, l_r=10e-3, reg_value=0.1, reg_type=0, loaddataset=None, epochs=20,
            loss_function="mae", series_df=pd.DataFrame(), isntance_moving_step=1, x_train_size=2, y_train_size=1):

        #super().fit(series_df, period, interval_minute, otimizar, output_variables, shift_steps)
        self.output_variables = output_variables
        self.series_df = series_df.sort_index()
        self.period = period
        self.interval_minute = interval_minute
        self.window_size = window_size #Tamanho dos subdatasets
        self.moving_step = moving_step #Tamanho da movinf average
        self.x_train_size = x_train_size # Historico de previsão
        self.y_train_size = y_train_size # Horizonte de previsão
        

        self.scalex=MinMaxScaler()
        self.scalex.fit(self.series_df.values)
        
        self.scaley=MinMaxScaler()
        self.scaley.fit(self.series_df[self.output_variables].values)
        
        if loaddataset is not None:
            
            self.series = self.loaddataset(loaddataset)
        else:
            self.series = self.split_dataset(window_size=self.window_size, moving_step=self.moving_step,
                                            x_train_size=self.x_train_size, y_train_size=self.y_train_size,
                                            split_dataset=self.test_perc, isntance_moving_step=isntance_moving_step)

        self.batch_size = batch_size

        print("Number of  datasets", len(self.series))
        print("Shape of x", self.series[0]["x_train"].shape)
        print("Shape of y", self.series[0]["y_train"].shape)

        print("Window x size", self.series[0]["x_train"].shape[1])
        print("Window y size",self.series[0]["y_train"].shape[1])
        regularizer = self.set_regulaztion(reg_type, reg_value)
        self.set_model(n_timesteps=self.series[0]["x_train"].shape[1],
                           n_features=self.series[0]["x_train"].shape[2],
                           n_outputs=self.series[0]["y_train"].shape[1], recurrent_regularizer=regularizer,
                           bias_regularizer=regularizer, l_r=l_r, loss_function=loss_function, dropout=0.2)
        
        last_loss=None
        
        for dataset in self.series:

            self.x_train = dataset["x_train"]
            self.y_train = dataset["y_train"]
            self.x_test = dataset["x_test"]
            self.y_test = dataset["y_test"]
            self.x_val = dataset["x_val"]
            self.y_val = dataset["y_val"]

            self.x_train, x_scaler = self.scale_data(data=self.x_train, scaler=self.scalex, fit=True)
            self.x_test,scaler = self.scale_data(data=self.x_test, scaler=self.scalex, fit=False)
            self.x_val,scaler = self.scale_data(data=self.x_val, scaler=self.scalex, fit=False)
     
            self.y_train, y_scaler = self.scale_data(data=self.y_train, scaler=self.scaley, fit=False, output=True)
            self.y_test,scaler = self.scale_data(data=self.y_test, scaler=self.scaley, fit=False, output=True)
            self.y_val,scaler = self.scale_data(data=self.y_val, scaler=self.scaley, fit=False, output=True)

            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode='min', verbose=1, patience=40)
            history = self.training(x_train=self.x_train, y_train=self.y_train, epochs=epochs, batch_size=self.batch_size,
                                    x_val=self.x_val, y_val=self.y_val, callback=es, stateful=False)
            
            self.history.append(history)
            yhat = self.model.predict(self.x_test, verbose=0)
            
            if last_loss is None or np.mean(history.history['loss'])<last_loss:
                last_loss=np.mean(history.history['loss'][-1])
            else:
                break
            
            

            
            scale_yhat = self.scaley.inverse_transform(yhat)
            scale_ytest = self.scaley.inverse_transform(self.y_test)
            
            self.getting_score(y_pred=scale_yhat, y_test=scale_ytest)
            self.y_test_y_hat.append([[scale_ytest], [scale_yhat]])


        new_dir = save_dir+"/"

        return self.mean_score(self.score_mae)
    
    def anomalies(self):
        dataset=self.create_instance_sub_dataset(sub_dataset=self.series_df.iloc[:], x_train_size=self.x_train_size,
                                                            y_train_size=0,
                                                            moving_step=1,
                                                            split_dataset=1, split_validation=1)
        self.x_train = dataset["x_train"]
        self.y_train = dataset["y_train"]
        self.x_test = dataset["x_test"]
        self.y_test = dataset["y_test"]
        self.x_val = dataset["x_val"]
        self.y_val = dataset["y_val"]
        self.x_train, x_scaler = self.scale_data(data=self.x_train, scaler=self.scalex, fit=False)
        pred = self.model.predict(self.x_train, verbose=0)
        scale_pred = self.scaley.inverse_transform(pred)
        return scale_pred
         

    def predict(self, day=False, predict=True):
        '''Convert days into time stampo'''
        if day:
            convert_ratio = DAY_MINUTE / self.interval_minute
            moving_step = int(convert_ratio * self.moving_step)
            x_len = int(convert_ratio * self.x_train_size)
            y_len = int(convert_ratio * self.y_train_size)
        train_len = len(self.series_df)

        final_df = self.series_df.copy()
        '''First x day are empty'''
        modeling_value = [0 for i in range(0, x_len)]
        index = self.series_df.iloc[:x_len].index.to_list()

        '''Part I - Predict train data (Modeling)'''
        for end_y_position in range(x_len+y_len, train_len+1, moving_step):
            begin_y_position = end_y_position - y_len
            begin_x_position = begin_y_position - x_len
            '''Use x datapoints to predict the next y datapoints'''
            x = np.expand_dims(self.series_df.iloc[begin_x_position:begin_y_position].to_numpy(), axis=0) #begin_x_postion = end_y_position
            y = self.series_df.loc[:, self.output_variables].iloc[begin_y_position:end_y_position]
            y_index = y.index.to_list()
            y = y.to_numpy()
            y_hat = self.model.predict(x)

            modeling_value += list(y_hat[0])
            index += y_index
        modeling = pd.DataFrame(data=modeling_value, index=pd.Index(index), columns=self.output_variables)

        for name_var in self.output_variables:
            final_df[name_var + "_model"] = modeling[name_var]

        '''Part II - Predicting'''
        if predict:

            x_list = self.series_df.iloc[train_len-x_len:train_len] #Get the needed value to train
            y_index = list()

            last_index = x_list.iloc[-1:].index.to_list()[0]#Last value of x
            delta = dt.timedelta(minutes=self.interval_minute)
            last_index = dt.datetime.strptime(last_index, DATEFORMAT) # LOCAL is needed

            '''Create the right index'''
            for j in range(0, y_len):
                index = last_index + delta
                y_index.append(index)
                last_index = index

            y_hat = self.model.predict(x)
            predicting_value = list(y_hat[0])

            predicted_df = pd.DataFrame(data=predicting_value, index=pd.DatetimeIndex(y_index),
                                        columns=self.output_variables) #Maybe watchout the columns parameter
            '''Merge predicted_df with final_df'''

            reindex = final_df.index.to_list() + predicted_df.index.to_list()
            final_df = final_df.reindex(reindex)
            for col in predicted_df.columns:
                final_df[col + "_predict"] = predicted_df[col]

        return final_df

    def mean_score(self, score_array):
        score_mean = 0
        for dataset_score in score_array:
            score_mean+=statistics.mean(dataset_score)
        return score_mean/(len(score_array))

    def split_dataset(self, day=False, window_size=15, moving_step=1, x_train_size=2, y_train_size=1,split_dataset=0.8,
                      isntance_moving_step=1):
        if day:
            convert_ratio = DAY_MINUTE / self.interval_minute
            window_size = int(convert_ratio*window_size)
            moving_step = int(convert_ratio*moving_step)
            x_train_size = int(convert_ratio*x_train_size)
            y_train_size = int(convert_ratio*y_train_size)
            isntance_moving_step = int(convert_ratio*isntance_moving_step)
        series = []
        dataset_len = len(self.series_df.iloc[:, 0]) #Number of datapoint
        print("window_size",window_size)
        print("moving_step",moving_step)
        for end_index in range(window_size, dataset_len+1, moving_step):
            begin_index = end_index - window_size
            sub_dataset = self.series_df.iloc[begin_index:end_index]
            
            series.append(self.create_instance_sub_dataset(sub_dataset=sub_dataset, x_train_size=x_train_size,
                                                           y_train_size=y_train_size,
                                                           moving_step=isntance_moving_step,
                                                           split_dataset=split_dataset, split_validation=0.8)) #Train size 2 dias assumindo que o interval é de 15 minutos
        return series

    def create_instance_sub_dataset(self, sub_dataset, x_train_size, y_train_size, moving_step,split_dataset,split_validation):
        x_data = []
        y_data = []
        
        
        for end_index in range(x_train_size + y_train_size, len(sub_dataset) + 1, moving_step):
            
            begin_index = end_index - (x_train_size + y_train_size)

            x_data.append(sub_dataset.iloc[begin_index:begin_index + x_train_size].to_numpy())
            y_data.append(sub_dataset.iloc[begin_index + x_train_size:end_index][self.output_variables].to_numpy().flatten())

        train_size = int(len(x_data)*split_dataset)

        x_train = x_data[:train_size] 
        y_train = y_data[:train_size]

        val_size = int(len(x_train)*split_validation)

        x_validation = x_train[val_size:len(x_train)]
        y__validation = y_train[val_size:len(y_train)]

        x_train = x_train[:val_size]
        y_train = y_train[:val_size]

        x_test = x_data[train_size:len(x_data)]
        y_test = y_data[train_size:len(y_data)]

        serie = {"x_train": np.array(x_train), "y_train": np.array(y_train),
                 "x_test": np.array(x_test),"y_test": np.array(y_test),
                 "x_val": np.array(x_validation),"y_val": np.array(y__validation),
                 }
        return serie

    def set_regulaztion(self, reg_type, reg_value):
        if reg_type == 0:
            return tf.keras.regularizers.l1(reg_value)
        if reg_type == 1:
            return tf.keras.regularizers.l2(reg_value)
        if reg_type == 2:
            return tf.keras.regularizers.l1_l2(l1=reg_value, l2=reg_value)

    def getting_score(self ,y_pred, y_test):
        #fig = plt.figure()
        mae = []
        mse = []
        for y_t, y_h in zip(y_test, y_pred):
            mae.append(mean_absolute_error(y_t, y_h).numpy())
            mse.append(mean_squared_error(y_t, y_h).numpy())
        self.score_mse.append(mse)
        self.score_mae.append(mae)

    def scale_data(self, data, scaler, fit=False, output=False):
        transform_data = []
        if not output:
            for row in data:
                if fit: scaler.fit(row)
                t_d = scaler.transform(row)
                transform_data.append(t_d)

            transform_data = np.array(transform_data)
        else:
            if fit: scaler.fit(data)
            transform_data = scaler.transform(data)
        return transform_data, scaler

    def set_model(self, n_timesteps, n_features, n_outputs, recurrent_regularizer, bias_regularizer, l_r, loss_function,
                  dropout=0):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.LSTM(n_outputs, activation='relu',
                                            input_shape=(n_timesteps, n_features),recurrent_dropout=dropout,
                                            recurrent_regularizer=bias_regularizer,
                                            kernel_regularizer=recurrent_regularizer,
                                            bias_regularizer=recurrent_regularizer))
        optimizer = tf.keras.optimizers.Adam(lr=l_r, clipnorm=0.7, clipvalue=0.3)   #0.7 0.3
        self.model.compile(loss=DICT_LOSS_FUNCTION.get(loss_function), optimizer=optimizer)
        self.model.summary()
    

    
    def training(self, x_train, y_train, epochs, batch_size, x_val, y_val, callback, stateful=False):
        if stateful:
            for i in range(0, epochs):
                history = self.model.fit(x_train, y_train, epochs=batch_size, batch_size=1,
                                         validation_data=(x_val, y_val), verbose=0, callback = callback)
                self.model.reset_states()
        else:
            history = self.model.fit(x_train,y_train, epochs=epochs, batch_size=batch_size,
                      validation_data=(x_val, y_val), verbose=0, callbacks=[callback])
        return history
        #self.plot_comparison(start_idx=10, length=1000, train=True)

    def save_dataset(self, dataset, filename):
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)
        try:
            with open(filename, 'w') as json_file:
                json.dump(dataset, json_file, cls=NumpyEncoder)
        except IOError:
            print("Saving dataset error")

    def save_dir(self, dir_name, dir_path=SAVE_MODEL_PATH, display=False):
        # Create dir
        if os.path.exists(dir_path+dir_name):
            shutil.rmtree(dir_path+dir_name)

        os.mkdir(dir_path+dir_name)
        new_dir = dir_path+dir_name+"/"

        # Save model and plot
        self.save_model(new_dir)
        

    def save_model(self, dir_):
        for index, dataset in enumerate(self.series):
            # Save sub_dataset
            self.save_dataset(dataset, dir_ + DATASET_FILE_NAME + str(index))
            # Save history
            history_file_name = HISTORY_FILE_NAME+str(index)
            history = self.history[index]
            with open(os.path.join(dir_, history_file_name), mode="w") as history_file:
                writer = csv.writer(history_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["loss", "val_loss"])
                for value in range(0, len(history.history["loss"])):
                    writer.writerow([history.history["loss"][value], history.history["val_loss"][value]])
            #Save score
            score_file_name = SCORE_FILE_NAME+str(index)
            with open(os.path.join(dir_, score_file_name), mode="w") as score_file:
                writer = csv.writer(score_file,  delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["mae","mse"])
                for value in range(0, len(self.score_mae[0])):
                    writer.writerow([self.score_mae[index][value], self.score_mse[index][value]])

        # Save model weight and parameters
        model_name = "x_size"+str(self.series[0]["x_train"].shape[1])+ \
                     "_subdataset_len_" + str(self.window_size)+\
                     "_y_size_"+str(self.series[0]["y_train"].shape[1])+"_date_"+str(dt.datetime.now())+\
                     "_interval_"+str(self.interval_minute)
        model_json = self.model.to_json()
        with open(os.path.join(dir_, "model.json"), "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(os.path.join(dir_, "model.h5"))



states = {'date2.start_date': '2018-12-14T00:00:00', 'date2.end_date': '2019-01-14T00:00:00',
 'cluster_paragens.value': ['saldanha'], 'paragens.value':
     ['406 - Praça Duque de Saldanha', '408 - Praça Duque de Saldanha', '407 - Praça Duque de Saldanha'],
 'agregar.value': ['agregar'], 'granularidade_em_minutos.value': '30', 'dias.value': ['todos_dias']
    , 'variaveis_input.value': ['check-in', 'check-out'], 'variavel_output.value': ['check-in'],
 'docas_ocupadas/livres.value': ['median'], 'contexto.value': ['nenhum'],
 'estacoes_meteorologicas.value': ['todas'], 'abordagens.value': ['machine_learning'],
 'modelo.value': 'lstm', 'janela_historico_(#pontos).value': '100',
 'horizonte_previsao_(#pontors).value': '10', 'ahead.value': [],
 'otimizar.value': [], 'bounds.value': ['envelope'],
 'avaliacao.value': 'rolling_CV_(Bayesian_optimization)',
 'parameterizacao.value': '<parameteros aqui>', 'estatisticas.value': []}

