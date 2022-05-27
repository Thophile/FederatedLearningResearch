import numpy as np
import pandas as pd
import datetime
from sklearn.utils import shuffle


def loadData(filePath):
    return pd.read_csv(filePath,
        engine='c',
        index_col=False,
        quoting=3,
        header=0,
        delimiter=",")


def LoadAll():
    fulldf = LoadOne(3)
    for i in ['4','5','6','7']:
        fulldf.append(LoadOne(i))
    return fulldf

def LoadOne(number):
    number = str(number)
    print('---------- Loading csv '+ number +' ----------')
    FloorDf = loadData(".\data\\2018Floor"+number+".csv")
    FloorDf.info()
    print("Floor 2018 OK")
    FloorDf.append(loadData(".\data\\2019Floor"+number+".csv"))
    FloorDf.info()
    print("Floor 2019OK")
    return FloorDf

def Wrangling(DF):
    print('---------- Wrangling ----------')
    for col in DF:
        if(DF.dtypes[col]=="float"):
            tempAverage = DF[col].mean()
            DF[col] = np.where(np.isnan(DF[col]), tempAverage, DF[col])

    DF["averageDegre"] = DF[['z1_S1(degC)', 'z2_S1(degC)', 'z4_S1(degC)', 'z5_S1(degC)']].mean(axis=1)
    DF.drop(['z1_S1(degC)', 'z2_S1(degC)', 'z4_S1(degC)', 'z5_S1(degC)'], axis=1, inplace=True)

    DF["averageHumidity"] = DF[['z1_S1(RH%)', 'z2_S1(RH%)', 'z4_S1(RH%)', 'z5_S1(RH%)']].mean(axis=1)
    DF.drop(['z1_S1(RH%)', 'z2_S1(RH%)', 'z4_S1(RH%)', 'z5_S1(RH%)'], axis=1, inplace=True)

    DF["averageLux"] = DF[['z1_S1(lux)', 'z2_S1(lux)', 'z4_S1(lux)', 'z5_S1(lux)']].mean(axis=1)
    DF.drop(['z1_S1(lux)', 'z2_S1(lux)', 'z4_S1(lux)', 'z5_S1(lux)'], axis=1, inplace=True)

    DF["Date"] = DF["Date"].map(lambda date: int(datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S').strftime('%H')) + int(datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S').strftime('%M'))/60)
    return DF

def onSplit(pandaDF):


    print("---------- Split ----------")

    #split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    #for train_index, test_index in split.split(pandaDF, pandaDF[["averageDegre", "averageHumidity", "averageLux"]]):
    #    strat_train_set = pandaDF.loc[train_index]
    #    strat_test_set = pandaDF.loc[test_index]

    #strat_train_label = strat_train_set[["averageDegre", "averageHumidity", "averageLux"]].copy()
    #strat_train_set = strat_train_set.drop(["averageDegre", "averageHumidity", "averageLux"], axis=1)
    
    #strat_test_label = strat_test_set[["averageDegre", "averageHumidity", "averageLux"]].copy()
    #strat_test_set = strat_test_set.drop(["averageDegre", "averageHumidity", "averageLux"], axis=1)
    
    pandaDF = shuffle(pandaDF)
    
    return pandaDF.drop(["averageDegre", "averageHumidity", "averageLux"], axis=1).to_numpy(), pandaDF[["averageDegre", "averageHumidity", "averageLux"]].copy().to_numpy()
