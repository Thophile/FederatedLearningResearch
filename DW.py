import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


def loadData(filePath, names):
    return pd.read_csv(filePath, names)


def LoadAll():
    batDf = loadData(".\data\batiments\batiment_62.csv", ["bnb_id", "altitude_sol", "geombui_area", "adedpe202006_logtype_min_classe_ener_ges", "adedpe202006_logtype_ratio_ges_conso", "adedpe202006_logtype_periode_construction", "adedpe202006_min_conso_ener", "adedpe202006_min_estim_ges", "adedpe202006_max_conso_ener", "adedpe202006_max_estim_ges", "anarnc202012_nb_log", "anarnc202012_nb_lot_tertiaire", "anarnc202012_nb_lot_tot", "mtedle2019_elec_conso_res_par_pdl_res", "mtedle2019_elec_nb_pdl_tot", "mtedle2019_gaz_conso_res_par_pdl_res", "mtedle2019_gaz_nb_pdl_tot", "adedpe202006_logtype_baie_fs", "adedpe202006_logtype_baie_mat", "adedpe202006_logtype_baie_orientation", "adedpe202006_logtype_baie_remplissage", "adedpe202006_logtype_baie_type_vitrage", "adedpe202006_logtype_ch_gen_lib_appoint", "adedpe202006_logtype_ch_gen_lib_princ", "adedpe202006_logtype_ch_is_solaire", "adedpe202006_logtype_ch_type_inst", "adedpe202006_logtype_ecs_gen_lib_princ", "adedpe202006_logtype_inertie", "adedpe202006_logtype_mur_ep_mat_ext", "adedpe202006_logtype_mur_mat_ext", "adedpe202006_logtype_mur_pos_isol_ext", "adedpe202006_logtype_pb_mat", "adedpe202006_logtype_pb_pos_isol", "adedpe202006_logtype_ph_pos_isol", "adedpe202006_logtype_presence_balcon", "adedpe202006_logtype_presence_climatisation", "adedpe202006_logtype_type_batiment", "adedpe202006_logtype_type_ventilation", "adedpe202006_nb_classe_ene_a", "adedpe202006_nb_classe_ene_b", "adedpe202006_nb_classe_ene_c", "adedpe202006_nb_classe_ene_d", "adedpe202006_nb_classe_ene_e", "adedpe202006_nb_classe_ene_f", "adedpe202006_nb_classe_ene_g", "adedpe202006_nb_classe_ene_nc", "cerffo2020_mat_mur_txt", "cerffo2020_mat_toit_txt", "igntop202103_bat_hauteur"])
    relAddBatDf= loadData(".\data\adresses\adresse_62.csv", ["bnb_id", "etaban202111_id"])
    adressDf= loadData(".\data\relAddBats\rel_adresse_batiment_opendata_62.csv", ["etaban202111_id", "etaban202111_latitude", "etaban202111_longitude"])
    temp = pd.merge(batDf, relAddBatDf, how="left", on=["bnb_id"])
    return pd.merge(temp, adressDf, how="left", on=["etaban202111_id"])





def onSplit(pandaDF):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(pandaDF, pandaDF["Attrition"]):
        strat_train_set = pandaDF.loc[train_index]
        strat_test_set = pandaDF.loc[test_index]

    strat_train_label = strat_train_set["Attrition"].copy()
    strat_train_set = strat_train_set.drop('Attrition', axis=1)
    
    strat_test_label = strat_test_set["Attrition"].copy()
    strat_test_set = strat_test_set.drop('Attrition', axis=1)

    return strat_train_set, strat_train_label, strat_test_set, strat_test_label