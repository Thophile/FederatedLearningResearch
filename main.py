import DW as DW
import EvaluatePrediction as PredictionEvaluator

import Modeles.RandomForest as RandomForestPredictor


pandaDf = DW.LoadAll()

x_train, y_train, x_test, y_test = DW.onSplit(pandaDf)


y_pred, y_pred_proba = RandomForestPredictor.RandomForestPredictor(x_train, y_train, x_test)
PredictionEvaluator.EvaluatePrediction(y_test, y_pred, y_pred_proba)