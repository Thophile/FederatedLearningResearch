from lib2to3.pytree import NodePattern
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
from feddist import roger_federer
import time

class Mode(Enum):
    GENERATE = 0
    GENERATE_ONE = 1
    FEDAVG = 2
    FEDDIST = 3
    TEST = 4
    FEDPER = 5

MODELS_DIRECTORY='./saved_models/4HL'
ITER_COUNT = 100000
HIDDEN_LAYER_SIZE = (50, 40, 30, 20)
MODEL_COUNT = 5
LOCAL_MODELS = []
MODE = Mode.FEDDIST

# Load the diabetes dataset
#X, y = datasets.load_diabetes(return_X_y=True)

# models values
mses = []
r2s = []

coefs_arr= []
intercepts_arr = []

#Load the buildings dataset
def getDF():
    pandaDf = DW.LoadOne(4)
    pandaDf = DW.Wrangling(pandaDf)
    X, y = DW.onSplit(pandaDf)
    return X, y

def combine(mat_list):
    avg=mat_list[0]
    i=1
    for mat in mat_list[1:]:
        avg += np.array(mat, dtype=object)
        i+=1
    return np.true_divide(avg, i).tolist()

def generate_model(X_train, y_train, max_iter=ITER_COUNT, fname=False, partial = False):
    model = MLPRegressor(random_state=1, hidden_layer_sizes=HIDDEN_LAYER_SIZE, max_iter=max_iter)
    if not partial:
        model.fit(X_train, y_train)
    else:
        model.partial_fit(X_train, y_train)

    if fname :
        dump(model, f'{MODELS_DIRECTORY}/{fname}.qtm')
    return model

if(MODE == Mode.GENERATE):
    X, y = getDF()

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
    X, y = getDF()
    # Bootstrap 1 models and saves it
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    model = generate_model(X_train, y_train, fname=f'model_4_4HL')
    y_pred = model.predict(X_test)
    mse, r2 = PredictionEvaluator.EvaluateReggression(y_test, y_pred)
    print("mse: %.2f" % (mse))
    print("r2: %.2f" % (r2))

elif(MODE == Mode.FEDAVG):
    X, y = getDF()

    for filename in os.listdir(MODELS_DIRECTORY):
        f = os.path.join(MODELS_DIRECTORY, filename)
        LOCAL_MODELS.append(load(f))

    start = time.time()

    for model in LOCAL_MODELS:
        # Export model neurons data
        coefs_arr.append(model.coefs_)
        intercepts_arr.append(model.intercepts_)
    
    # Agregate
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    federated_mlp = generate_model(X_train, y_train, partial=True)
    federated_mlp.coefs_ = combine(coefs_arr)
    federated_mlp.intercepts_ = combine(intercepts_arr)

    print("---------- Fedavg results ----------")
    print(f"Federated in {time.time() - start}s")
    y_pred = federated_mlp.predict(X_test)

    (mse, mae, r2) = PredictionEvaluator.EvaluateReggression(y_test, y_pred)
    print("mse: %.2f" % (mse))
    print("mae: %.2f" % (mae))
    print("r2: %.2f" % (r2))

elif(MODE == Mode.FEDDIST):
    X, y = getDF()
    NEW_LAYER_SIZE = list(HIDDEN_LAYER_SIZE)

    for filename in os.listdir(MODELS_DIRECTORY):
        f = os.path.join(MODELS_DIRECTORY, filename)
        LOCAL_MODELS.append(load(f))

    start = time.time()
    NEW_LAYER_SIZE, new_coefs, new_intercepts = roger_federer(LOCAL_MODELS, HIDDEN_LAYER_SIZE)

    print("---------- Creating global model with size of " + str(NEW_LAYER_SIZE) + " ----------")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    model = MLPRegressor(random_state=1, hidden_layer_sizes=tuple(NEW_LAYER_SIZE), max_iter=ITER_COUNT)
    model.partial_fit(X_train, y_train)
    model.coefs_ = new_coefs
    model.intercepts_ = new_intercepts

    print("---------- Fedpdist results ----------")
    print(f"Federated in {time.time() - start}s")
    y_pred = model.predict(X_test)

    (mse, mae, r2) = PredictionEvaluator.EvaluateReggression(y_test, y_pred)
    print("mse: %.2f" % (mse))
    print("mae: %.2f" % (mae))
    print("r2: %.2f" % (r2))

elif(MODE == Mode.TEST):
    print("TEST")

elif(MODE == Mode.FEDPER):
    for filename in os.listdir(MODELS_DIRECTORY):
        f = os.path.join(MODELS_DIRECTORY, filename)
        LOCAL_MODELS.append(load(f))

    X, y = getDF()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    start = time.time()
    for model in LOCAL_MODELS:
        # Export model neurons data
        coefs_arr.append(model.coefs_)
        intercepts_arr.append(model.intercepts_)

    multi_layer_neuron=[]
    coefs_averaged=[]
    node_averaged=[]
    all_intercepts_avg=[]
    for i in range(0,len(LOCAL_MODELS[0].coefs_)-2):
        layer_intercepts_avg=[]
        coefs_averaged=[]
        for y in range(0,len(LOCAL_MODELS[0].coefs_[i])):
            node_averaged=[]
            for z in range (0,len(LOCAL_MODELS[0].coefs_[i][y])):
                node_path=0
                for ii in range (0,len(LOCAL_MODELS)):
                    node_path =node_path+LOCAL_MODELS[ii].coefs_[i][y][z]
                node_averaged.append(node_path/len(LOCAL_MODELS))
            coefs_averaged.append(node_averaged)
        coefs_averaged = np.array(coefs_averaged,dtype=object)
        multi_layer_neuron.append(coefs_averaged)
        for w in range(0,len(LOCAL_MODELS[0].intercepts_[i])):
            intercepts_avg=0
            for ii in range (0,len(LOCAL_MODELS)):
                    intercepts_avg =intercepts_avg+LOCAL_MODELS[ii].intercepts_[i][w]
            layer_intercepts_avg.append(intercepts_avg/len(LOCAL_MODELS))
        all_intercepts_avg.append(np.array(layer_intercepts_avg,dtype=object))

    federated_mlp = generate_model(X_train, y_train, partial=True)
    for i in range (0,len(all_intercepts_avg)-1):
        federated_mlp.intercepts_[i]=np.array(all_intercepts_avg[i],dtype=object)
    for i in range (0,len(multi_layer_neuron)-1):
        federated_mlp.coefs_[i]=np.array(multi_layer_neuron[i],dtype=object)
    for coef in federated_mlp.intercepts_:
        print(len(coef))

    print("---------- Fedper results ----------")
    print(f"Federated in {time.time() - start}s")
    y_pred = federated_mlp.predict(X_test)

    (mse, mae, r2) = PredictionEvaluator.EvaluateReggression(y_test, y_pred)
    print("mse: %.2f" % (mse))
    print("mae: %.2f" % (mae))
    print("r2: %.2f" % (r2))
