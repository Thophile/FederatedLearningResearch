from statistics import mode
from Modeles.MLP import *
import EvaluatePrediction as PredictionEvaluator
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np

ITER_COUNT = 10000
MODEL_COUNT = 5

def combine(mat_list):
    avg=mat_list[0]
    i=1
    for mat in mat_list[1:]:
        avg += np.array(mat, dtype=object)
        i+=1
    return np.true_divide(avg, i).tolist()

def bootstrap_model(max_iter=ITER_COUNT):
    X_train, X_test, y_train, y_test = train_test_split(diabetes_X, diabetes_y, test_size=0.33)
    model = MLPRegressor(random_state=1, max_iter=max_iter)
    model.fit(X_train, y_train)
    return model, X_test, y_test


# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

mses = []
r2s = []

coefs_arr= []
intercepts_arr = []

for i in range(MODEL_COUNT):
    # Bootstrap 5 models and saves the 5 trained params
    model, X_test, y_test = bootstrap_model()
    y_pred = model.predict(X_test)

    # Evaluate model
    mse, r2 = PredictionEvaluator.EvaluateReggression(y_test, y_pred)
    mses.append(mse)
    r2s.append(r2)

    # Export model neurons data
    coefs_arr.append(model.coefs_)
    intercepts_arr.append(model.intercepts_)

print("---------- Normal models ----------")
print("Mean mse: %.2f" % (sum(mses) / len(mses)))
print("Mean r2: %.2f" % (sum(r2s) / len(r2s)))

federated_mlp, X_test, y_test = bootstrap_model()
federated_mlp.coefs_ = combine(coefs_arr)
federated_mlp.intercepts_ = combine(intercepts_arr)
y_pred = federated_mlp.predict(X_test)

print("---------- Federated learning ----------")
federated_mse, federated_r2 = PredictionEvaluator.EvaluateReggression(y_test, y_pred)
print("Federated mse: %.2f" % (federated_mse))
print("Federated r2: %.2f" % (federated_r2))

combinediter_mlp, X_test, y_test = bootstrap_model(MODEL_COUNT*ITER_COUNT)
y_pred = combinediter_mlp.predict(X_test)

print("---------- Combined iter learning ----------")
combinediter_mse, combinediter_r2 = PredictionEvaluator.EvaluateReggression(y_test, y_pred)
print("Combined iter mse: %.2f" % (combinediter_mse))
print("Combined iter r2: %.2f" % (combinediter_r2))








