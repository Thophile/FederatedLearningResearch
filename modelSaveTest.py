from Modeles.MLP import *
import EvaluatePrediction as PredictionEvaluator
from sklearn.model_selection import train_test_split
from sklearn import datasets

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

mses = []
r2s = []

for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(diabetes_X, diabetes_y, test_size=0.33)

    model, y_pred = MlpRegressor(X_train, y_train, X_test, i=i)
    print(model)
    mse, r2 = PredictionEvaluator.EvaluateReggression(y_test, y_pred)
    mses.append(mse)
    r2s.append(r2)
print("--------------------------")
print("Mean mse: %.2f" % (sum(mses) / len(mses)))
print("Mean r2: %.2f" % (sum(r2s) / len(r2s)))

