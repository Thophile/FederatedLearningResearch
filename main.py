import DW as DW
import EvaluatePrediction as PredictionEvaluator

import Modeles.RandomForest as RandomForestPredictor
import Modeles.GradientDecent as Grad

pandaDf = DW.LoadOne(3)
pandaDf = DW.Wrangling(pandaDf)

pandaDf.info()
print(pandaDf.head())


X, y = DW.onSplit(pandaDf)




#y_pred, y_pred_proba = RandomForestPredictor.RandomForestPredictor(x_train, y_train, x_test)
#y_pred, y_pred_proba = Grad.GradientDescentPredictor(x_train, y_train, x_test)
#PredictionEvaluator.EvaluatePrediction(y_test, y_pred, y_pred_proba)