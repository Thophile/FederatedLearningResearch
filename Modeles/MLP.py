from sklearn.neural_network import MLPRegressor
from joblib import dump, load

def MlpRegressor(X_train, y_train, X_test, i=-1):
    Mlp = MLPRegressor(random_state=1, max_iter=5000)
    Mlp.fit(X_train, y_train)
    y_pred = Mlp.predict(X_test)
    
    dump(Mlp, f'data/Mlp{i if i != -1 else ""}.joblib')
    return Mlp, y_pred