from email import header
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
import re

from sklearn.utils import shuffle
from sklearn import preprocessing


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
    print('--------------------------------------------Loading csv '+ number +'---------------------------------------')
    FloorDf = loadData(".\data\\2018Floor"+number+".csv")
    FloorDf.info()
    print("Floor 2018 OK")
    FloorDf.append(loadData(".\data\\2019Floor"+number+".csv"))
    FloorDf.info()
    print("Floor 2019OK")
    return FloorDf

def Wrangling(DF):
    print('--------------------------------------------Wrangling---------------------------------------')
    for col in DF:
        if(DF.dtypes[col]=="float"):
            tempAverage = DF[col].mean()
            DF[col] = np.where(np.isnan(DF[col]), tempAverage, DF[col])

    DF["averageDegre"] = DF[['z1_S1(degC)', 'z2_S1(degC)', 'z4_S1(degC)', 'z5_S1(degC)']].mean(axis=1)
    DF["averageHumidity"] = DF[['z1_S1(RH%)', 'z2_S1(RH%)', 'z4_S1(RH%)', 'z5_S1(RH%)']].mean(axis=1)
    DF["averageLux"] = DF[['z1_S1(lux)', 'z2_S1(lux)', 'z4_S1(lux)', 'z5_S1(lux)']].mean(axis=1)

    return DF


 

 


def onSplit(pandaDF):


    print("---------------------------------------------Split--------------------------------------")

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

def x_y_split(pandaDF, limit=False):
    print("---------------------------------------------Split--------------------------------------")

    pandaDF.drop("bnb_id", axis=1, inplace=True)
    #pandaDF.drop("etaban202111_id", axis=1, inplace=True)

    pandaDFD = pandaDF.copy()
    
    le = preprocessing.LabelEncoder()

    pandaDF_2 = pandaDF.apply(le.fit_transform)

      
    pandaDF_2["mtedle2019_elec_conso_tot"] = pd.cut(pandaDF_2["mtedle2019_elec_conso_tot"], 32, labels=np.arange(32))

    enc = preprocessing.OneHotEncoder()
    enc.fit(pandaDF_2)

    transformed = enc.transform(pandaDF_2)

    oheDF = pd.DataFrame(transformed)

    for categorie in pandaDFD.columns:

        pandaDFD[categorie] = pd.concat([pandaDF[categorie], oheDF], axis=1).drop(categorie, axis=1)

    if limit:
        pandaDF_2 = pandaDF_2[:limit]
    print(pandaDF_2.mtedle2019_elec_conso_tot[100])
    return pandaDF_2.drop('mtedle2019_elec_conso_tot', axis=1).to_numpy(), pandaDF_2["mtedle2019_elec_conso_tot"].copy().to_numpy()