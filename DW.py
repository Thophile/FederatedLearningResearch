from email import header
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import preprocessing


def loadData(filePath, names, cols):
    return pd.read_csv(filePath,
        engine='c',
        index_col=False,
        usecols=cols, 
        names=names,
        quoting=3,
        header=0,
        delimiter=";")


def LoadAll():
    batDf = loadData(".\data\\batiments\\batiment_85.csv", ["mtedle2019_elec_conso_tot", "bnb_id", "altitude_sol", "geombui_area", "adedpe202006_logtype_min_classe_ener_ges", "adedpe202006_logtype_ratio_ges_conso", "adedpe202006_logtype_periode_construction", "adedpe202006_min_conso_ener", "adedpe202006_min_estim_ges", "adedpe202006_max_conso_ener", "adedpe202006_max_estim_ges", "anarnc202012_nb_log", "anarnc202012_nb_lot_tertiaire", "anarnc202012_nb_lot_tot", "mtedle2019_elec_conso_res_par_pdl_res", "mtedle2019_elec_nb_pdl_tot", "mtedle2019_gaz_conso_res_par_pdl_res", "mtedle2019_gaz_nb_pdl_tot", "adedpe202006_logtype_baie_fs", "adedpe202006_logtype_baie_mat", "adedpe202006_logtype_baie_orientation", "adedpe202006_logtype_baie_remplissage", "adedpe202006_logtype_baie_type_vitrage", "adedpe202006_logtype_ch_gen_lib_appoint", "adedpe202006_logtype_ch_gen_lib_princ", "adedpe202006_logtype_ch_is_solaire", "adedpe202006_logtype_ch_type_inst", "adedpe202006_logtype_ecs_gen_lib_princ", "adedpe202006_logtype_inertie", "adedpe202006_logtype_mur_ep_mat_ext", "adedpe202006_logtype_mur_mat_ext", "adedpe202006_logtype_mur_pos_isol_ext", "adedpe202006_logtype_pb_mat", "adedpe202006_logtype_pb_pos_isol", "adedpe202006_logtype_ph_pos_isol", "adedpe202006_logtype_presence_balcon", "adedpe202006_logtype_presence_climatisation", "adedpe202006_logtype_type_batiment", "adedpe202006_logtype_type_ventilation", "adedpe202006_nb_classe_ene_a", "adedpe202006_nb_classe_ene_b", "adedpe202006_nb_classe_ene_c", "adedpe202006_nb_classe_ene_d", "adedpe202006_nb_classe_ene_e", "adedpe202006_nb_classe_ene_f", "adedpe202006_nb_classe_ene_g", "adedpe202006_nb_classe_ene_nc", "cerffo2020_mat_mur_txt", "cerffo2020_mat_toit_txt", "igntop202103_bat_hauteur"], ["mtedle2019_elec_conso_tot", "bnb_id", "altitude_sol", "geombui_area", "adedpe202006_logtype_min_classe_ener_ges", "adedpe202006_logtype_ratio_ges_conso", "adedpe202006_logtype_periode_construction", "adedpe202006_min_conso_ener", "adedpe202006_min_estim_ges", "adedpe202006_max_conso_ener", "adedpe202006_max_estim_ges", "anarnc202012_nb_log", "anarnc202012_nb_lot_tertiaire", "anarnc202012_nb_lot_tot", "mtedle2019_elec_conso_res_par_pdl_res", "mtedle2019_elec_nb_pdl_tot", "mtedle2019_gaz_conso_res_par_pdl_res", "mtedle2019_gaz_nb_pdl_tot", "adedpe202006_logtype_baie_fs", "adedpe202006_logtype_baie_mat", "adedpe202006_logtype_baie_orientation", "adedpe202006_logtype_baie_remplissage", "adedpe202006_logtype_baie_type_vitrage", "adedpe202006_logtype_ch_gen_lib_appoint", "adedpe202006_logtype_ch_gen_lib_princ", "adedpe202006_logtype_ch_is_solaire", "adedpe202006_logtype_ch_type_inst", "adedpe202006_logtype_ecs_gen_lib_princ", "adedpe202006_logtype_inertie", "adedpe202006_logtype_mur_ep_mat_ext", "adedpe202006_logtype_mur_mat_ext", "adedpe202006_logtype_mur_pos_isol_ext", "adedpe202006_logtype_pb_mat", "adedpe202006_logtype_pb_pos_isol", "adedpe202006_logtype_ph_pos_isol", "adedpe202006_logtype_presence_balcon", "adedpe202006_logtype_presence_climatisation", "adedpe202006_logtype_type_batiment", "adedpe202006_logtype_type_ventilation", "adedpe202006_nb_classe_ene_a", "adedpe202006_nb_classe_ene_b", "adedpe202006_nb_classe_ene_c", "adedpe202006_nb_classe_ene_d", "adedpe202006_nb_classe_ene_e", "adedpe202006_nb_classe_ene_f", "adedpe202006_nb_classe_ene_g", "adedpe202006_nb_classe_ene_nc", "cerffo2020_mat_mur_txt", "cerffo2020_mat_toit_txt", "igntop202103_bat_hauteur"])
    print("Bat OK")
    relAddBatDf= loadData(".\data\\relAddBats\\rel_adresse_batiment_opendata_85.csv", ["bnb_id", "etaban202111_id"], ["bnb_id", "etaban202111_id"])
    print("Rel OK")
    adressDf= loadData(".\data\\adresses\\adresse_85.csv", ["etaban202111_id", "etaban202111_latitude", "etaban202111_longitude"], ["etaban202111_id", "etaban202111_latitude", "etaban202111_longitude"])
    print("Adresse OK")
    temp = pd.merge(batDf, relAddBatDf, how="left", on=["bnb_id"])

    temp2 = pd.merge(temp, adressDf, how="left", on=["etaban202111_id"])
    return batDf





def onSplit(pandaDF):


    print("---------------------------------------------Split--------------------------------------")

    pandaDF.drop("bnb_id", axis=1, inplace=True)
    #pandaDF.drop("etaban202111_id", axis=1, inplace=True)

    pandaDFD = pandaDF.copy()

    

    le = preprocessing.LabelEncoder()

    pandaDF_2 = pandaDF.apply(le.fit_transform)
    print(pandaDF_2.head())
    pandaDF_2.info()
      
    pandaDF_2["mtedle2019_elec_conso_tot"] = pd.cut(pandaDF_2["mtedle2019_elec_conso_tot"], 32, labels=np.arange(32))
    print(pandaDF_2.head())
    enc = preprocessing.OneHotEncoder()
    enc.fit(pandaDF_2)

    transformed = enc.transform(pandaDF_2)

    oheDF = pd.DataFrame(transformed)

    for categorie in pandaDFD.columns:

        pandaDFD[categorie] = pd.concat([pandaDF[categorie], oheDF], axis=1).drop(categorie, axis=1)

    pandaDFD.info()
    #print(pandaDFD.head())

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(pandaDF_2, pandaDF_2["mtedle2019_elec_conso_tot"]):
        strat_train_set = pandaDF_2.loc[train_index]
        strat_test_set = pandaDF_2.loc[test_index]

    strat_train_label = strat_train_set["mtedle2019_elec_conso_tot"].copy()
    strat_train_set = strat_train_set.drop('mtedle2019_elec_conso_tot', axis=1)
    
    strat_test_label = strat_test_set["mtedle2019_elec_conso_tot"].copy()
    strat_test_set = strat_test_set.drop('mtedle2019_elec_conso_tot', axis=1)

    return strat_train_set, strat_train_label, strat_test_set, strat_test_label