# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:05:13 2022

@author: Calvin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 11:42:43 2022

@author: Calvin
"""

from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
import matplotlib.pyplot as plt

class EDA():
    def __init__(self):
        pass
    
    def plot_graph(self,df):
        '''
        

        Parameters
        ----------
        df : TYPE
            Plotting 4 different graphs.

        Returns
        -------
        None.

        '''
        plt.figure()
        plt.plot(df['cases_0_4'])
        plt.plot(df['cases_12_17'])
        plt.plot(df['cases_18_29'])
        plt.plot(df['cases_30_39'])
        plt.plot(df['cases_40_49'])
        plt.plot(df['cases_50_59'])
        plt.plot(df['cases_60_69'])
        plt.plot(df['cases_70_79'])
        plt.plot(df['cases_80'])
        plt.legend(['cases_0_4','cases_5_11','cases_12_17','cases_18_29',
                    'cases_30_39','cases_40_49','cases_50_59','cases_60_69',
                    'cases_70_79','cases_80',])
        plt.show()
        
        plt.figure()
        plt.plot(df['cases_new'])
        plt.plot(df['cases_import'])
        plt.plot(df['cases_recovered'])
        plt.plot(df['cases_active'])
        plt.plot(df['cases_cluster'])
        plt.plot(df['cases_unvax'])
        plt.plot(df['cases_pvax'])
        plt.plot(df['cases_fvax'])
        plt.plot(df['cases_boost'])
        plt.legend(['cases_new','cases_import','cases_recovered',
                    'cases_active','cases_cluster','cases_unvax',
                    'cases_pvax','cases_fvax','cases_boost'])
        plt.show()
        
        plt.figure()
        plt.plot(df['cases_child'])
        plt.plot(df['cases_adolescent'])
        plt.plot(df['cases_adult'])
        plt.plot(df['cases_elderly'])
        plt.legend(['cases_child','cases_adolescent','cases_adult',
                    'cases_elderly'])
        plt.show()
        
        plt.figure()
        plt.plot(df['cluster_import'])
        plt.plot(df['cluster_religious'])
        plt.plot(df['cluster_community'])
        plt.plot(df['cluster_highRisk'])
        plt.plot(df['cluster_education'])
        plt.plot(df['cluster_detentionCentre'])
        plt.plot(df['cluster_workplace'])
        plt.legend(['cluster_import','cluster_religious','cluster_community',
                    'cluster_highRisk','cluster_education',
                    'cluster_detentionCentre','cluster_workplace'])
        plt.show()
        
class ModelCreation():
    def __init__(self):
            pass
    def simple_lstm_layer(self,X_train,num_node=64,dropout_rate=0.2,
                          output_node=1):
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1:])))
        model.add(LSTM(num_node,return_sequences=(True)))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(num_node))
        model.add(Dropout(dropout_rate))
        model.add(Dense(output_node,activation='relu'))
        model.summary()
        
        return model
