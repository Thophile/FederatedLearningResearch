from Modeles.MLP import *
import EvaluatePrediction as PredictionEvaluator
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np

def combine(mat_list):
    avg = False
    for mat in mat_list:
        if avg:
            avg += mat
            i+= 1
        else:
            avg=mat
            i=1
    return np.true_divide(np.array(avg, dtype=object), i).tolist()

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

mses = []
r2s = []

coefs_arr= []
intercepts_arr = []

for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(diabetes_X, diabetes_y, test_size=0.33)

    model, y_pred = MlpRegressor(X_train, y_train, X_test, i=i)
    coefs_arr.append(model.coefs_)
    intercepts_arr.append(model.intercepts_)
    mse, r2 = PredictionEvaluator.EvaluateReggression(y_test, y_pred)
    mses.append(mse)
    r2s.append(r2)
print("--------------------------")
print("Mean mse: %.2f" % (sum(mses) / len(mses)))
print("Mean r2: %.2f" % (sum(r2s) / len(r2s)))

X_train, X_test, y_train, y_test = train_test_split(diabetes_X, diabetes_y, test_size=0.33)
federated_mlp = MLPRegressor(random_state=1, max_iter=5000)
federated_mlp.fit(X_train, y_train)
federated_mlp.coefs_ = combine(coefs_arr)
federated_mlp.intercepts_ = combine(intercepts_arr)

y_pred = federated_mlp.predict(X_test)
dump(federated_mlp, 'data/Federated_mlp.joblib')
print("---------- Federated learning ----------")
federated_mse, federated_r2 = PredictionEvaluator.EvaluateReggression(y_test, y_pred)






