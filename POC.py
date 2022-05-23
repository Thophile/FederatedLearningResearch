from statistics import mode
from Modeles.MLP import *
import EvaluatePrediction as PredictionEvaluator
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from joblib import dump, load
import os
from enum import Enum
import DW

class Mode(Enum):
    GENERATE = 0
    GENERATE_ONE = 1
    LOAD = 2

MODELS_DIRECTORY='./saved_models'
ITER_COUNT = 10000
MODEL_COUNT = 5
LOCAL_MODELS = []
MODE = Mode.LOAD

# Load the diabetes dataset
#X, y = datasets.load_diabetes(return_X_y=True)

#Load the buildings dataset
pandaDf = DW.LoadAll()
#pandaDf.info()
X, y= DW.x_y_split(pandaDf, limit=100)

#print(len(y))

# models values
mses = []
r2s = []

coefs_arr= []
intercepts_arr = []

def combine(mat_list):
    avg=mat_list[0]
    i=1
    for mat in mat_list[1:]:
        avg += np.array(mat, dtype=object)
        i+=1
    return np.true_divide(avg, i).tolist()

def generate_model(X_train, y_train, max_iter=ITER_COUNT, fname=False, partial = False):
    model = MLPRegressor(random_state=1, max_iter=max_iter)
    if not partial:
        model.fit(X_train, y_train)
    else:
        model.partial_fit(X_train, y_train)

    if fname :
        dump(model, f'{MODELS_DIRECTORY}/{fname}.qtm')
    return model

if(MODE == Mode.GENERATE):
    for i in range(MODEL_COUNT):
        # Bootstrap 5 models and saves the 5 trained params
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        model = generate_model(X_train, y_train, fname=f'model{i}')
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

    # Build fl model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    federated_mlp = generate_model(X_train, y_train)
    federated_mlp.coefs_ = combine(coefs_arr)
    federated_mlp.intercepts_ = combine(intercepts_arr)
    y_pred = federated_mlp.predict(X_test)

    print("---------- Federated learning ----------")
    federated_mse, federated_r2 = PredictionEvaluator.EvaluateReggression(y_test, y_pred)
    print("Federated mse: %.2f" % (federated_mse))
    print("Federated r2: %.2f" % (federated_r2))

    #Build combined iteration model to compare with fl model on a same iteration basis
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    combined_iter_mlp = generate_model(X_train, y_train, MODEL_COUNT*ITER_COUNT)
    y_pred = combined_iter_mlp.predict(X_test)

    print("---------- Combined iter learning ----------")
    combined_iter_mse, combined_iter_r2 = PredictionEvaluator.EvaluateReggression(y_test, y_pred)
    print("Combined iter mse: %.2f" % (combined_iter_mse))
    print("Combined iter r2: %.2f" % (combined_iter_r2))

elif(MODE == Mode.GENERATE_ONE):
    # Bootstrap 1 models and saves it
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    model = generate_model(X_train, y_train, fname=f'model_name')


elif(MODE == Mode.LOAD):
    for filename in os.listdir(MODELS_DIRECTORY):
        f = os.path.join(MODELS_DIRECTORY, filename)
        LOCAL_MODELS.append(load(f))

    for model in LOCAL_MODELS:
        # Export model neurons data
        coefs_arr.append(model.coefs_)
        intercepts_arr.append(model.intercepts_)
    
    # Agregate
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    federated_mlp = generate_model(X_train, y_train, partial=True)
    federated_mlp.coefs_ = combine(coefs_arr)
    federated_mlp.intercepts_ = combine(intercepts_arr)
    y_pred = federated_mlp.predict(X_test)

    print("---------- Federated learning ----------")
    federated_mse, federated_r2 = PredictionEvaluator.EvaluateReggression(y_test, y_pred)
    print("Federated mse: %.2f" % (federated_mse))
    print("Federated r2: %.2f" % (federated_r2))