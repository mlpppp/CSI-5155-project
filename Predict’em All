# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:41:17 2019

@author: funrr
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import KFold
    
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler  
    
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
    
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from skrules import SkopeRules 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier

os.chdir(r"C:\Users\funrr\Desktop\ML project")
work_dir= os.getcwd()




def ReadData():
#read dataset into dataframe
    datas_dir= os.getcwd() + '\\300k_csv'
    data = pd.read_csv(datas_dir +'\\300k.csv', header = 0)
    return data

def CheckMissing(data):
#check if there is any nan in data set
    missing_tag = 0
    for (columnName, columnData) in data.iteritems():    
        nullidx = data.index[data.isnull()[columnName] == True].tolist()
        if len(nullidx) != 0: 
            print('Missing Colunm Name : ', columnName)
            print('Missing idx: ')
            print(*nullidx, sep = ", ")  
            missing_tag = 1

    if missing_tag == 0:
        print('No missing value found')
        
     
        
def Aux_DataDistVis(data):
    ax =data['city'].hist(bins = 98, grid=False,xlabelsize=3.5, xrot = 90)
    ax.set_title('#appear by cities')
    ax.set_xlabel("cities")
    ax.set_ylabel("#appear")
    fig = ax.get_figure()
    fig.savefig(r'C:\Users\funrr\Desktop\ML project\plots' + '\\city' +'.jpg', dpi = 600)
    

    ax = data.city.value_counts().plot(kind='bar', fontsize = 4)
    ax.set_title('#appear by cities')
    ax.set_xlabel("cities")
    ax.set_ylabel("#appear")
    fig = ax.get_figure()
    fig.savefig(r'C:\Users\funrr\Desktop\ML project\plots' + '\\city' +'.jpg', dpi = 600)
    
    Ame = data[data['continent'] == 'America']
    Euro = data[data['continent'] == 'Europe']
    Asia = data[data['continent'] == 'Asia']
    
    ax1 = Ame['rarity'].value_counts().plot(kind='bar', fontsize = 10, grid=True)
    ax1.set_title('rarity class in America')
    ax1.set_ylabel("#appearing")
    ax1.set_xlabel("rarity class")
    fig1 = ax1.get_figure()
    fig1.savefig(r'C:\Users\funrr\Desktop\ML project\plots' + '\\AM' +'.jpg', dpi = 600)
    
    ax2 =Euro['rarity'].value_counts().plot(kind='bar', fontsize = 10, grid=True)
    ax2.set_title('rarity class in Europe')
    ax2.set_ylabel("#appearing")
    ax2.set_xlabel("rarity class")
    fig2 = ax2.get_figure()
    fig2.savefig(r'C:\Users\funrr\Desktop\ML project\plots' + '\\EU' +'.jpg', dpi = 600)
    
    ax3 =Asia['rarity'].value_counts().plot(kind='bar', fontsize = 10, grid=True)
    ax3.set_title('rarity class in Asia')
    ax3.set_ylabel("#appearing")
    ax3.set_xlabel("rarity class")
    fig3 = ax3.get_figure()
    fig3.savefig(r'C:\Users\funrr\Desktop\ML project\plots' + '\\AS' +'.jpg', dpi = 600)
    
    
def MappingRare(data):
#Adding rarity feature to the data
    df = pd.read_excel(r'C:\Users\funrr\Desktop\ML project\temp\rare.xlsx', header= None)
    a = df.loc[0:9, 0].values
    b = df.loc[0:39, 2].values
    c = df.loc[0:80, 4].values
    d = df.loc[0:9, 6].values
    e = df.loc[0:9, 8].values
    rare_class = np.zeros(data.shape[0])
    PokId = data['pokemonId'].values
    for idx, ID in enumerate(PokId):
        if ID in a:
            rare_class[idx] = 1
        elif ID in b:
            rare_class[idx] = 2
        elif ID in c:
            rare_class[idx] = 3
        elif ID in d:
            rare_class[idx] = 4
        elif ID in e:
            rare_class[idx] = 5
    if not 0 in rare_class:
        data['rarity'] = rare_class
        return data
    else:
        print('error')
        return

#def SubDataVis(data):
#explore features in each rarity class
#    common = data.loc[data['rarity'] == 1].copy()     
#    uncommon = data.loc[data['rarity'] == 2].copy() 
#    rare = data.loc[data['rarity'] == 3].copy() 
#    var_rare = data.loc[data['rarity'] == 4].copy()
#    sup_rare = data.loc[data['rarity'] == 5].copy()
    

    
def FeaturePreDrop(data):
#drop features that are obviously redundant
    #kill some redundant features
    drop = ['city', 'pokemonId', 'class','appearedLocalTime','_id','cellId_90m','cellId_180m','cellId_370m','cellId_730m','cellId_1460m','cellId_2920m','cellId_5850m','appearedDay','appearedMonth','appearedYear','sunriseMinutesMidnight','sunriseHour','sunriseMinute','sunriseMinutesSince','sunsetMinutesMidnight','sunsetHour','sunsetMinute','sunsetMinutesBefore', 'weatherIcon','appearedTimeOfDay']  
    data.drop(columns=drop, inplace = True)
    #kill missing values(samples) in 'appearedDayOfWeek'
    data = data[data.appearedDayOfWeek != "dummy_day"]
    return data
    
 
def DisplayFeatures(data):
#display all features name
    for i in range(data.shape[1]):
        print(data.columns[i])    
        
def FeatureEng(data): 
    #1.
    #feature transformation for 'appearedDayOfWeek'
    #based on 'https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/'
    data.replace({'appearedDayOfWeek': {'Sunday': 7., 'Saturday': 6., 'Friday': 5., 'Thursday': 4., 'Wednesday': 3., 'Tuesday': 2., 'Monday': 1.}}, inplace= True)
    data['sin_weekday'] = np.sin(np.array((2*np.pi*data.appearedDayOfWeek/7).values, dtype=np.float32))
    data['cos_weekday'] = np.cos(np.array((2*np.pi*data.appearedDayOfWeek/7).values, dtype=np.float32))
    data.drop('appearedDayOfWeek', axis=1, inplace=True)
    #2.
    #feature transformation for 'appearedHour' and 'appearedMinute'
    #same as 'weekdays', by seconds past 0-o'clock
    data['secOfday'] = (data.appearedHour*60+data.appearedMinute)*60 
    data.drop('appearedHour', axis=1, inplace=True)
    data.drop('appearedMinute', axis=1, inplace=True)
    data['sin_secOfday'] = np.sin(np.array((2*np.pi*data.secOfday/(24*60*60)).values, dtype=np.float32))
    data['cos_secOfday'] = np.cos(np.array((2*np.pi*data.secOfday/(24*60*60)).values, dtype=np.float32))
    data.drop('secOfday', axis=1, inplace=True)
    
    
    #3.
    #feature transformation for 'longitude' 
    #same as 'weekdays', by degrees [-pi, +pi]

    data['sin_longitude'] = np.sin(np.array((2*np.pi*data.longitude/360).values, dtype=np.float32))
    data['cos_longitude'] = np.cos(np.array((2*np.pi*data.longitude/360).values, dtype=np.float32))
    data.drop('longitude', axis=1, inplace=True)

    #4.
    #convert 'weather' to numerics
    Rain = np.zeros(data.shape[0])
    Cloud = np.zeros(data.shape[0])
    Clear = np.zeros(data.shape[0])
    Wind = np.zeros(data.shape[0])
    Humid = np.zeros(data.shape[0])
    
    Rain[(data['weather'] == 'Drizzle').values] = 1
    Rain[(data['weather'] == 'DrizzleandBreezy' ).values] = 1    
    Rain[(data['weather'] == 'LightRain' ).values] = 2   
    Rain[(data['weather'] == 'LightRainandBreezy').values] = 2 
    Rain[(data['weather'] == 'Rain').values] = 3
    Rain[(data['weather'] == 'RainandWindy' ).values] = 3
    Rain[(data['weather'] == 'HeavyRain').values] = 4
    ####
    Cloud[(data['weather'] == 'PartlyCloudy').values] = 1
    Cloud[(data['weather'] == 'HumidandPartlyCloudy' ).values] = 1    
    Cloud[(data['weather'] == 'DryandPartlyCloudy' ).values] = 1   
    Cloud[(data['weather'] == 'WindyandPartlyCloudy').values] = 1 
    Cloud[(data['weather'] == 'BreezyandPartlyCloudy').values] = 1
    
    Cloud[(data['weather'] == 'MostlyCloudy'  ).values] = 2
    Cloud[(data['weather'] == 'DryandMostlyCloudy' ).values] = 2
    Cloud[(data['weather'] == 'BreezyandMostlyCloudy').values] = 2
   
    Cloud[(data['weather'] == 'Overcast').values] = 3
    Cloud[(data['weather'] == 'HumidandOvercast'  ).values] = 3
    Cloud[(data['weather'] == 'BreezyandOvercast').values] = 3
    ###
    Clear[(data['weather'] == 'Clear'  ).values] = 1
    Clear[(data['weather'] == 'Foggy' ).values] = 2    
    Clear[(data['weather'] == 'WindyandFoggy' ).values] = 2 
    ###
    Wind[(data['weather'] == 'Breezy').values] = 1
    Wind[(data['weather'] == 'LightRainandBreezy' ).values] = 1    
    Wind[(data['weather'] == 'DrizzleandBreezy' ).values] = 1   
    Wind[(data['weather'] == 'BreezyandOvercast').values] = 1 
    Wind[(data['weather'] == 'BreezyandPartlyCloudy').values] = 1
    Wind[(data['weather'] == 'BreezyandMostlyCloudy').values] = 1

    Wind[(data['weather'] == 'Windy'  ).values] = 2
    Wind[(data['weather'] == 'RainandWindy' ).values] = 2
    Wind[(data['weather'] == 'WindyandPartlyCloudy' ).values] = 2
    Wind[(data['weather'] == 'WindyandFoggy' ).values] = 2
 
    Wind[(data['weather'] == 'DangerouslyWindy').values] = 3
    ###
    Humid[(data['weather'] == 'Dry').values] = 1
    Humid[(data['weather'] == 'DryandMostlyCloudy' ).values] = 1    
    Humid[(data['weather'] == 'DryandPartlyCloudy' ).values] = 1   
    Humid[(data['weather'] == 'Humid').values] = 2 
    Humid[(data['weather'] == 'HumidandOvercast').values] = 2
    Humid[(data['weather'] == 'HumidandPartlyCloudy').values] = 2
    
    data['Rain'] = Rain
    data['Cloud'] = Cloud
    data['Clear'] = Clear
    data['Wind'] = Wind
    data['Humid'] = Humid
    data.drop('weather', axis=1, inplace=True)
    #5.
    #convert 151 cooc to 5 coor
    df = pd.read_excel(r'C:\Users\funrr\Desktop\ML project\temp\rare.xlsx', header= None)
    a = df.loc[0:9, 0].values
    b = df.loc[0:39, 2].values
    c = df.loc[0:80, 4].values
    d = df.loc[0:9, 6].values
    e = df.loc[0:9, 8].values
    
    cooc_r = np.zeros([data.shape[0],5], dtype=bool)

    
    for i in range(151):
        pid = 'cooc_'+ str(i+1)
        real = (data[pid] == 1).values #get a array: sample true if its cooc_i ==1
        if (i+1) in a:                  #find the rarity class #i belongs to
            r = 0
        elif (i+1) in b:
            r = 1
        elif (i+1) in c:
            r = 2
        elif (i+1) in d:
            r = 3
        elif (i+1) in e:
            r = 4
        cooc_r[:, r] =  cooc_r[:, r] | real  #use OR operhand to add coor_i to its respective cooc_r class
        
    for i in range(5):              #add the new cooc_rarity to the data set
        s = 'cooc_rarity_' + str(i+1)
        data[s] = cooc_r[:,i]
    
    for i in range(151):
        s = 'cooc_'+ str(i+1)
        data.drop(s, axis=1, inplace=True)
        

    
    #6. keep only "Europe', 'America' and 'Asia', and use one-hot encode


    America = (data['continent'] == 'America').values
    Europe = (data['continent'] == 'Europe').values
    Asia = (data['continent'] == 'Asia').values
    
    keep = America |Europe |Asia
    data['cont_America'] = America
    data['cont_Europe'] = Europe
    data['cont_Asia'] = Asia
    
    data = data[keep]
    data.drop('continent',axis=1, inplace=True)
    
    
            
          
    
    #7.fix problem of 'pokestopDistanceKm' (The feature is a mixing of '?', numeric string and number), and convert feature to float64
    
         
    #'?' in s   #return True
    data = data[data.pokestopDistanceKm != "?"]
    
    s=data.pokestopDistanceKm.values 
     
    for idx, item in enumerate(s):
       if type(s[idx]) is not float:
           if '?' in s[idx]:
               print(s[idx])
           
    for idx, item in enumerate(s):
       if type(s[idx]) is not float:
           s[idx] = float(s[idx])
         
    data['pokestopDistanceKm'] = s
    data = data.astype('float64')
    
    #8 Normalizing remaining features (MIN_MAX normalization)
    
    #'latitude'
    c = (data['latitude'].values - data['latitude'].min())/(data['latitude'].max()-data['latitude'].min())
    data['latitude'] = c
    
    #'temperature'
    c = (data['temperature'].values - data['temperature'].min())/(data['temperature'].max()-data['temperature'].min())
    data['temperature'] = c
    
    #'windSpeed'
    c = (data['windSpeed'].values - data['windSpeed'].min())/(data['windSpeed'].max()-data['windSpeed'].min())
    data['windSpeed'] = c
    
    #'windBearing'
    c = (data['windBearing'].values - data['windBearing'].min())/(data['windBearing'].max()-data['windBearing'].min())
    data['windBearing'] = c
    
    #'pressure'
    c = (data['pressure'].values - data['pressure'].min())/(data['pressure'].max()-data['pressure'].min())
    data['pressure'] = c    
    
    #'population_density'
    c = (data['population_density'].values - data['population_density'].min())/(data['population_density'].max()-data['population_density'].min())
    data['population_density'] = c  
    
    #'pokestopDistanceKm' and 'gymDistanceKm'
    #the two distance are at most 375KM, but almost all data are in range[0,6], so data point out of the range will be dropped
    #after limiting the range, we do not do normalization
    data = data[data['gymDistanceKm'] <=6]
    data = data[data['pokestopDistanceKm'] <=6]

    #9 one-hot terrainType
    for i in data.terrainType.unique():
        s = 'terrain_'+str(i)
        data[s] = (data.terrainType == i)
    data.drop('terrainType',axis=1, inplace=True) 
    

        
    #10.
    #convert all True/Flase to 1/0
    data.replace(True,1, inplace=True)
    data.replace(False,0, inplace=True)  
    
    #11 move the class to predict(rarity) to the front
    column = ['rarity', 'latitude', 'closeToWater', 'temperature', 'windSpeed', 'windBearing',
       'pressure', 'population_density', 'urban', 'suburban', 'midurban',
       'rural', 'gymDistanceKm', 'gymIn100m', 'gymIn250m', 'gymIn500m',
       'gymIn1000m', 'gymIn2500m', 'gymIn5000m', 'pokestopDistanceKm',
       'pokestopIn100m', 'pokestopIn250m', 'pokestopIn500m', 'pokestopIn1000m',
       'pokestopIn2500m', 'pokestopIn5000m', 'sin_weekday',
       'cos_weekday', 'sin_secOfday', 'cos_secOfday', 'sin_longitude',
       'cos_longitude', 'Rain', 'Cloud', 'Clear', 'Wind', 'Humid',
       'cooc_rarity_1', 'cooc_rarity_2', 'cooc_rarity_3', 'cooc_rarity_4',
       'cooc_rarity_5', 'cont_America', 'cont_Europe', 'cont_Asia',
       'terrain_14.0', 'terrain_13.0', 'terrain_10.0', 'terrain_12.0',
       'terrain_1.0', 'terrain_7.0', 'terrain_5.0', 'terrain_0.0',
       'terrain_8.0', 'terrain_2.0', 'terrain_4.0', 'terrain_11.0',
       'terrain_16.0', 'terrain_9.0']
    data = data[column]
    
    
    return data
#print("Skewness: %f" % df['Sales'].skew())
#print("Kurtosis: %f" % df['Sales'].kurt())
def npData5c(data):
#return 5 classes features and labels
    data_np = data.to_numpy()
    X = data_np[:,1:]
    y = data_np[:,0]
    return X, y
def PCA(X):
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(X) 
    plt.plot(pca.singular_values_)
    plt.set_title('Singular Values of X')
    plt.set_ylabel("features rank")
    plt.set_xlabel("Singular Values")
    plt.savefig(r'C:\Users\funrr\Desktop\ML project\plots' + '\\PCA' +'.jpg', dpi = 600)
    #pca = PCA(n_components = 55)
    #pca.fit(X) 
    #X = pca.transform(X)
    
    

def binarizeData(data):
#turn data set from 5 classes to 2 classes
    d2 = data.copy()
    rare_2class = np.zeros(d2.shape[0])
    rare_5class = d2['rarity'].values
    for idx, rare in enumerate(rare_5class):
        if rare <= 3:
            rare_2class[idx] = 0
        else:
            rare_2class[idx] = 1
    d2['rarity'] = rare_2class
    return d2


def npData2c(data):
#return 2 classes features and labels
    dataa = binarizeData(data)
    data_np = dataa.to_numpy()
    X = data_np[:,1:]
    y = data_np[:,0]
    return X, y

def randSample(X, y , size = 20000):
#randomly sample (uniform distribution) #size sample from data set
    if size > X.shape[0]:
        print('too much sample')
        return
    sample = np.random.randint(0, high = X.shape[0], size = size)
    X = X[sample]
    y = y[sample]
    return X, y 





def script():
#script that generate transformed data set 
    data = ReadData()
    data = MappingRare(data)
    data = FeaturePreDrop(data)
    data = FeatureEng(data)
    X, y = npData5c(data)
    #X, y = npData2c(data)
    #X, y = randSample(X, y , size = 20000)
    return

def Toytest(X, y):
#test the effect of samping before and after splitting train and test set
    acc_5c = []         
    confusion_mat_5c = []
    precision_5c = []
    recall_5c = []
    fscore_5c = []


    kf = KFold(n_splits=10, shuffle = True) 
    


#train without sampling   
    P = np.zeros([10,2])
    R = np.zeros([10,2])
    F = np.zeros([10,2])
    from sklearn.tree import DecisionTreeClassifier
    a = 0
    p,r,f = np.zeros([5]),np.zeros([5]),np.zeros([5])
    c = np.zeros([5,5])
    i = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        tree = DecisionTreeClassifier()             ####################
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
    
        a += accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
        c +=  confusion_matrix(y_test, y_pred)
        prf = precision_recall_fscore_support(y_test, y_pred)
        p += prf[0]
        r += prf[1]
        f += prf[2]
        P[i][0], P[i][1]= prf[0][3],prf[0][4]
        R[i][0], R[i][1]= prf[1][3],prf[1][4]
        F[i][0], F[i][1]= prf[2][3],prf[2][4]
        
        i +=1
        print(accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))
        print(confusion_matrix(y_test, y_pred)) 
        print(precision_recall_fscore_support(y_test, y_pred))
        
    acc_5c.append(a/10)
    confusion_mat_5c.append(c/10)
    precision_5c.append(p/10) 
    recall_5c.append(r/10)
    fscore_5c.append(f/10)    
    
    
    
    
#sample before split
    rus_balance = RandomUnderSampler(sampling_strategy = {1:10000,2:10000, 3:10000})
    sm_balance = SMOTE(sampling_strategy = {4:10000, 5:5000})
    X_sample, y_sample = rus_balance.fit_resample(X, y)
    X_sample, y_sample = sm_balance.fit_resample(X_sample, y_sample)

 
    from sklearn.tree import DecisionTreeClassifier
    a = 0
    p,r,f = np.zeros([5]),np.zeros([5]),np.zeros([5])
    c = np.zeros([5,5])
    for train_index, test_index in kf.split(X_sample):
        X_train, X_test = X_sample[train_index], X_sample[test_index]
        y_train, y_test = y_sample[train_index], y_sample[test_index]
        tree = DecisionTreeClassifier()             ####################
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
    
        a += accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
        c +=  confusion_matrix(y_test, y_pred)
        prf = precision_recall_fscore_support(y_test, y_pred)
        p += prf[0]
        r += prf[1]
        f += prf[2]
    
        print(accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))
        print(confusion_matrix(y_test, y_pred)) 
        print(precision_recall_fscore_support(y_test, y_pred))
        
    acc_5c.append(a/10)
    confusion_mat_5c.append(c/10)
    precision_5c.append(p/10) 
    recall_5c.append(r/10)
    fscore_5c.append(f/10)
    
#############sample after split    
    from sklearn.tree import DecisionTreeClassifier
    a = 0
    p,r,f = np.zeros([5]),np.zeros([5]),np.zeros([5])
    c = np.zeros([5,5])
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        X_sample, y_sample = rus_balance.fit_resample(X_train, y_train)
        X_sample, y_sample = sm_balance.fit_resample(X_sample, y_sample)   
        
        tree = DecisionTreeClassifier()             ####################
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
    
        a += accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
        c +=  confusion_matrix(y_test, y_pred)
        prf = precision_recall_fscore_support(y_test, y_pred)
        p += prf[0]
        r += prf[1]
        f += prf[2]
    
        print(accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))
        print(confusion_matrix(y_test, y_pred)) 
        print(precision_recall_fscore_support(y_test, y_pred))
        
        
    acc_5c.append(a/10)
    confusion_mat_5c.append(c/10)
    precision_5c.append(p/10) 
    recall_5c.append(r/10)
    fscore_5c.append(f/10)
    
#############    
    
    return F,P,R 


def testKNN(data):
#test the best k in KNN to use
    X, y = npData5c(data)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    rus_balance = RandomUnderSampler(sampling_strategy = {1:10000,2:10000, 3:10000})
    sm_balance = SMOTE(sampling_strategy = {4:10000, 5:5000})    
    for k in range(2,9):
        X_train, y_train = rus_balance.fit_resample(X_train, y_train)
        X_train, y_train = sm_balance.fit_resample(X_train, y_train)       
        clf = KNeighborsClassifier( n_neighbors = k )      
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))
        print(confusion_matrix(y_test, y_pred)) 
        print(precision_recall_fscore_support(y_test, y_pred)) 


def Sample5c(X, y, model):
#use sampling method to rebalance data and train each model
#model in ['SVC', 'tree', 'kNN', 'rule', 'RF', 'Adaboost']

    
    if model not in ['SVC', 'tree', 'kNN', 'rule', 'RF', 'Adaboost']:
        print('model not support')
        return
        

    acc_5c = []                     #store average result(10 folds)
    confusion_mat_5c = []
    precision_5c = []
    recall_5c = []
    fscore_5c = []

    precision_all =[]               #store raw score for class 4 and 5(1o for each model)
    recall_all = []  
    fscore_all = []
    
    kf = KFold(n_splits=10, shuffle = True)   

    

    
# only down sample majority class(1,2,3,4,)   
    a = 0
    p,r,f = np.zeros([5]),np.zeros([5]),np.zeros([5])
    c = np.zeros([5,5])
    P = np.zeros([10,2])
    R = np.zeros([10,2])
    F = np.zeros([10,2])
    rus_balance = RandomUnderSampler(sampling_strategy = {1:2500,2:2500, 3:2500,4:2500})

    i = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        X_train, y_train = rus_balance.fit_resample(X_train, y_train)
 
        if model == 'SVC':
            clf = SVC(kernel = 'rbf', gamma='scale')        ####################
        elif model == 'tree':
            clf = DecisionTreeClassifier()             ####################
        elif model == 'kNN': 
            clf = KNeighborsClassifier()             ####################
        elif model == 'rule' :
            rule = SkopeRules()
            clf = OneVsRestClassifier(rule)

        elif model == 'RF' :
            clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        elif model == 'Adaboost' :
            clf = AdaBoostClassifier(n_estimators=100, random_state=0)
        
     
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        a += accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
        c +=  confusion_matrix(y_test, y_pred)
        prf = precision_recall_fscore_support(y_test, y_pred)
        p += prf[0]
        r += prf[1]
        f += prf[2]
        P[i][0], P[i][1]= prf[0][3],prf[0][4]
        R[i][0], R[i][1]= prf[1][3],prf[1][4]
        F[i][0], F[i][1]= prf[2][3],prf[2][4]      
        i+= 1         
        print(accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))
        print(confusion_matrix(y_test, y_pred)) 
        print(precision_recall_fscore_support(y_test, y_pred)) 
    
    acc_5c.append(a/10)
    confusion_mat_5c.append(c/10)
    precision_5c.append(p/10) 
    recall_5c.append(r/10)
    fscore_5c.append(f/10)
    precision_all.append(P)
    recall_all.append(R)
    fscore_all.append(F)
    ##################

#balanced sampling, oversample 4 and 5 
    a = 0
    p,r,f = np.zeros([5]),np.zeros([5]),np.zeros([5])
    c = np.zeros([5,5])
    P = np.zeros([10,2])
    R = np.zeros([10,2])
    F = np.zeros([10,2])    
    
    rus_balance = RandomUnderSampler(sampling_strategy = {1:10000,2:10000, 3:10000})
    sm_balance = SMOTE(sampling_strategy = {4:10000, 5:5000})
    i = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        X_train, y_train = rus_balance.fit_resample(X_train, y_train)
        X_train, y_train = sm_balance.fit_resample(X_train, y_train)
        
        if model == 'SVC':
            clf = SVC(kernel = 'rbf', gamma='scale')        ####################
        elif model == 'tree':
            clf = DecisionTreeClassifier()             ####################
        elif model == 'kNN': 
            clf = KNeighborsClassifier()             ####################
        elif model == 'rule' :
            rule = SkopeRules()
            clf = OneVsRestClassifier(rule)

        elif model == 'RF' :
            clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        elif model == 'Adaboost' :
            clf = AdaBoostClassifier(n_estimators=100, random_state=0)
        
        
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        a += accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
        c +=  confusion_matrix(y_test, y_pred)
        prf = precision_recall_fscore_support(y_test, y_pred)
        p += prf[0]
        r += prf[1]
        f += prf[2]
        P[i][0], P[i][1]= prf[0][3],prf[0][4]
        R[i][0], R[i][1]= prf[1][3],prf[1][4]
        F[i][0], F[i][1]= prf[2][3],prf[2][4]   
        i+= 1               
        print(accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))
        print(confusion_matrix(y_test, y_pred)) 
        print(precision_recall_fscore_support(y_test, y_pred)) 
    
    acc_5c.append(a/10)
    confusion_mat_5c.append(c/10)
    precision_5c.append(p/10) 
    recall_5c.append(r/10)
    fscore_5c.append(f/10)
    precision_all.append(P)
    recall_all.append(R)
    fscore_all.append(F)
    
    
    
    return (acc_5c,confusion_mat_5c,precision_5c,recall_5c,fscore_5c, [precision_all,recall_all,fscore_all])
    

def Sample2c(data, model):
#use sampling method to rebalance data and train each model
#model in ['SVC', 'tree', 'kNN', 'rule', 'RF', 'Adaboost']
    from sklearn.model_selection import KFold
    
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler  
    
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_recall_fscore_support
    
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from skrules import SkopeRules 
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.multiclass import OneVsRestClassifier
    
    if model not in ['SVC', 'tree', 'kNN', 'rule', 'RF', 'Adaboost']:
        print('model not support')
        return
        
    X, y = npData2c(data)
    acc_5c = []                     #store average result(10 folds)
    confusion_mat_5c = []
    precision_5c = []
    recall_5c = []
    fscore_5c = []

    precision_all =[]               #store raw score for class 4 and 5(1o for each model)
    recall_all = []  
    fscore_all = []
    
    kf = KFold(n_splits=10, shuffle = True)   

    

    


 
#balanced sampling, oversample 4 and 5 
    a = 0
    p,r,f = np.zeros([2]),np.zeros([2]),np.zeros([2])
    c = np.zeros([2,2])
    P = np.zeros([10,2])
    R = np.zeros([10,2])
    F = np.zeros([10,2])
    
    rus_balance = RandomUnderSampler(sampling_strategy = {0:30000})
    sm_balance = SMOTE(sampling_strategy = {1:15000})
    i = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        X_train, y_train = rus_balance.fit_resample(X_train, y_train)
        X_train, y_train = sm_balance.fit_resample(X_train, y_train)
        
        if model == 'SVC':
            clf = SVC(kernel = 'rbf', gamma='scale')        ####################
        elif model == 'tree':
            clf = DecisionTreeClassifier()             ####################
        elif model == 'kNN': 
            clf = KNeighborsClassifier()             ####################
        elif model == 'rule' :
            rule = SkopeRules()
            clf = OneVsRestClassifier(rule)

        elif model == 'RF' :
            clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        elif model == 'Adaboost' :
            clf = AdaBoostClassifier(n_estimators=100, random_state=0)
        
        
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        a += accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)
        c +=  confusion_matrix(y_test, y_pred)
        prf = precision_recall_fscore_support(y_test, y_pred)
        p += prf[0]
        r += prf[1]
        f += prf[2]
        P[i][0], P[i][1]= prf[0][0],prf[0][1]
        R[i][0], R[i][1]= prf[1][0],prf[1][1]
        F[i][0], F[i][1]= prf[2][0],prf[2][1]   
        i+= 1               
        print(accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))
        print(confusion_matrix(y_test, y_pred)) 
        print(precision_recall_fscore_support(y_test, y_pred)) 
    
    acc_5c.append(a/10)
    confusion_mat_5c.append(c/10)
    precision_5c.append(p/10) 
    recall_5c.append(r/10)
    fscore_5c.append(f/10)
    precision_all.append(P)
    recall_all.append(R)
    fscore_all.append(F)
    
    
    
    return (acc_5c,confusion_mat_5c,precision_5c,recall_5c,fscore_5c, [precision_all,recall_all,fscore_all])




def oneClass(data):
#test one-class learning
    from sklearn.svm import OneClassSVM

    X, y = npData2c(data)

    acc = []
    CM = []
    prf = []
    
    
     
    kf = KFold(n_splits=10, shuffle = True) 

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train,y_train = randSample(X_train,y_train, size = 50000)
        
        clf = OneClassSVM(gamma='auto').fit(X_train)    
        y_pred = clf.predict(X_test)
        yy_pred = np.zeros(y_pred.shape)
        
        for i in range(y_pred.shape[0]):
            if y_pred[i] == -1:
                yy_pred[i] = y_pred[i]+2
            elif y_pred[i] == 1:
                yy_pred[i] = y_pred[i]-1
            
        acc.append(accuracy_score(y_test, yy_pred, normalize=True, sample_weight=None))
        CM.append(confusion_matrix(y_test, yy_pred))
        prf.append(precision_recall_fscore_support(y_test, yy_pred))

        print(accuracy_score(y_test, yy_pred, normalize=True, sample_weight=None))
        print(confusion_matrix(y_test, yy_pred)) 
        print(precision_recall_fscore_support(y_test, yy_pred))
        
        c = np.zeros([2,2])
        p,r,f = np.zeros([2]),np.zeros([2]),np.zeros([2])   
        for i in range(10):
            p += prf[i][0]
            r += prf[i][1]
            f += prf[i][2]
            c += CM[i]
        c = c/10  
        p = p/10
        r = r/10
        f = f/10
        
    return acc,CM, prf
        
