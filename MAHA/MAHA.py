#!/usr/bin/env python
# coding: utf-8

# ## Importing The Libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error

import warnings
warnings.filterwarnings('ignore')


# ## The main Class

# In[49]:


class MAHA:
    
    ''' Constructor '''
    def __init__(self):
        None
    
    ''' Model Finding'''
    def model_calc(self, df):
        modes = []
        temp = list(df.columns)
        print(temp)
        for i in temp:
            y = df[i]
            x = df.drop(i, axis = 1)

            if df[i].dtype == 'int64' or df[i].dtype == 'float64':
                lr = LinearRegression(normalize = True)
                lr.fit(x, y)
                modes.append(lr)

            elif df[i].dtype == 'category' or df[i].dtype == 'object':
                lg = LogisticRegression()
                lg.fit(x, y)
                modes.append(lg)

        return modes
    
    ''' Splitting DataFrame '''
    def splitDataFrame(self, df):
        unclean = df[df.isnull().any(axis = 1) == True]
        clean = df[df.isnull().any(axis = 1) == False]
        return clean, unclean
    
    ''' Finding Index Column '''
    def indexColDetector(self, df):
        t1 = sum(range(df.shape[0] + 1))
        t2 = sum(range(df.shape[0]))

        for i in list(df.columns):
            if df[i].dtype == 'int64' or df[i].dtype == 'float64':
                vals = list(df[i])
                s = sum(vals)
                if t1 == s or t2 == s: df.drop(i, axis = 1, inplace = True)

        return df
    
    ''' Dropping Unnecesary Columns '''
    def dropColumns(self, df, obj_drop = 0.7, drop_cols = 0.6):
        dummy = df.copy()
        dummy = self.indexColDetector(dummy)
        print(dummy.columns)

        for i in list(dummy.columns):
            num = dummy[i].nunique()
            if num == 1 or (num >= obj_drop * df.shape[0] and df[i].dtype == 'object'):
                dummy = dummy.drop(i, axis = 1)

        for i in list(dummy.columns):
            c = dummy[i].isnull().sum()
            if c/df.shape[0] > drop_cols:
                dummy.drop(i, axis = 1, inplace = True)

        return dummy
    
    ''' Replace Mean and Mode of Columns'''
    def replaceMeanMode(self, df):
        for i in list(df.columns):
            check = df[i].dtypes

            if check == 'int64' or check == 'float64':
                a = df[i].astype('float').mean(axis = 0)
                df[i].fillna(a, inplace = True)

            elif check == 'object' or check == 'category':
                df[i].fillna(df[i].mode()[0], inplace = True)

        return df
    
    ''' Finding Mean and Mode of Columns'''
    def findMeanMode(self, df):
        meanMode = []
        for i in list(df.columns):
            check = df[i].dtypes

            if check == 'int64' or check == 'float64':
                a = df[i].astype('float').mean(axis = 0)
                meanMode.append(a)

            elif check == 'object' or check == 'category':
                meanMode.append(df[i].mode()[0])

        return meanMode
    
    ''' The Main Function '''
    def MAHA(self, df, obj_drop = 0.7, drop_cols = 0.6):
        df = self.dropColumns(df, obj_drop, drop_cols)
        cols = list(df.columns)

        meanMode = self.findMeanMode(df)
        cl, ucl = self.splitDataFrame(df)
        models = self.model_calc(cl)

        newFrame = pd.DataFrame()

        for i, n in enumerate(cols):
            dummy = ucl.copy()

            c = cols[0:i]
            p = meanMode[0:i]
            for j, k in zip(p, c):
                dummy[k].fillna(j, inplace = True)

            c = cols[i+1:]
            p = meanMode[i+1:]
            for j, k in zip(p, c):
                dummy[k].fillna(j, inplace = True)

            _, xy = self.splitDataFrame(dummy)
            z = xy[xy == 'NaN'].index

            if not xy.empty:
                x = xy.drop(n, axis = 1)
                y = xy[n]

                pred = models[i].predict(x)

                k = 0
                for f in range(df.shape[0]):
                    for g in range(z.shape[0]):
                        if f == z[g]:
                            df.iloc[f, i] = pred[k]
                            k = k + 1

        return df

