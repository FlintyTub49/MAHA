#!/usr/bin/env python
# coding: utf-8

# ## Importing The Libraries

# In[7]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression

import warnings
warnings.filterwarnings('ignore')


# ## The main Class

# In[8]:


class MetaClean:
    '''
    This class will contain all the functions required for one stop data cleaning.

    NOTE:
    All the columns have to be converted to the desired datatype before using any function from MAHA. The function does not change any dataype of any column.
    '''
    
    ''' Constructor '''
    def __init__(self):
        None
    
    ''' Model Finding'''
    def model_calc(self, df):
        '''
        Calculates which model to be used for all the columns in the dataframe. 
        
        This creates a model on the input dataset. If the variable/column is of the datatype float it performs Linear Regression on the variable/column. If the variable/column is of the datatype int/category and has 2 unique values it performs Logistic Regression on the variable/column. 
        
        :type df: pandas.core.frame.DataFrame
        :param df: The dataframe on whose columns the models have to be built.
        '''    
    
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
        '''
        This function splits the dataframe into clean and unclean parts wrt NA values.
        
        This function checks for NA or Null values present in a dataframe and assigns those rows to a separate dataframe. The same is done for the case where NA or Null values are not present in a dataframe, The not-Null values are stored in another datadrame.
        
        :type df: pandas.core.frame.DataFrame
        :param df: The dataframe which is to be split into clean and unclean dataframes.
        '''
        
        unclean = df[df.isnull().any(axis = 1) == True]
        clean = df[df.isnull().any(axis = 1) == False]
        return clean, unclean
    
    ''' Finding Index Column '''
    def indexColDetector(self, df):
        '''
        This function detects the index colum and drops it.
        
        This function checks for the Sum of rows (as range(rows + 1)) in a dataframe and the sum of the values of a column if they are similar. If yes, the column is dropped.
        
        :type df: pandas.core.frame.DataFrame
        :param df: The dataframe from which index column are to be detected and dropped.
        '''
        
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
        '''
        This function determines which columns are to be dropped.
        
        This function checks if there are over 70% (of rows) unique values and/or over 60% Null values and/or only 1 unique value in an entire dataframe.
        
        :type df: pandas.core.frame.DataFrame
        :param df: The dataframe from which columns are to be dropped.
        
        :type obj_drop: float
        :param obj_drop: The ratio of unique object values to number of rows above which a column has to be dropped.
        
        :type drop_cols: float
        :param drop_cols: The ratio of Null values to number of rows above which a column has to be dropped.
        '''
        
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
    
    ''' Label Encoding Appropriate Columns '''
    def label(self, df):
        '''
        This function automatically label encodes appropriate columns.
        
        :type df: pandas.core.frame.DataFrame
        :param df: The dataframe from which columns are to be dropped.'''
        
        le = LabelEncoder()
        for col in list(df.columns):
            if df[col].dtypes == 'object':
                index = ~df[col].isna()
                df.loc[index, col] = le.fit_transform(df.loc[index, col]) 
                df[col]=df[col].astype('category')
        return df
    
    ''' Replace Mean and Mode of Columns'''
    def replaceMeanMode(self, df):
        '''
        This function replaces the NA values with Mean or Mode, depending on the type of the variable passed on.
        
        This function checks the datatype if it is float or int/object to replace with mean and mode respectively.
        
        :type df: pandas.core.frame.DataFrame
        :param df: The dataframe where NA/Null values are to be replaced.
        '''
        
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
        '''
        This function finds the mean and mode of the variables/columns of the dataframe
        
        :type df: pandas.core.frame.DataFrame
        :param df: The dataframe where mean/mode of columns are to be found.
        '''
        
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
    def MAHA(self, df, obj_drop = 0.7, drop_cols = 0.6, scale = False):
        '''
        This function is the main function which calls all the functions to provide one line cleaning of a dataset.
        
        :type df: pandas.core.frame.DataFrame
        :param df: The dataframe which is to be cleaned.
        
        :type obj_drop: float
        :param obj_drop: The ratio of unique object values to number of rows above which a column has to be dropped.
        
        :type drop_cols: float
        :param drop_cols: The ratio of Null values to number of rows above which a column has to be dropped.
        
        :type scale: boolean
        :param scale: If the dataset has to be scaled or not.
        '''
        
        df = self.dropColumns(df, obj_drop, drop_cols)
        cols = list(df.columns)
        df = self.label(df)

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
        
        if scale:
            sc = StandardScaler()
            df = pd.DataFrame(sc.fit_transform(df))
            
        return df

if __name__ == '__main__':
    None

