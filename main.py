import DW as DW
import EvaluatePrediction as PredictionEvaluator



pandaDf = DW.LoadAll()

x_train, y_train, x_test, y_test = DW.onSplit(pandaDf)