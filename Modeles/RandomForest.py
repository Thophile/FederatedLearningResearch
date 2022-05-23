from sklearn.ensemble import RandomForestClassifier

def RandomForestPredictor(X_train, y_train, X_test):
    print('----------------------------Random Forest------------------------------------')
    RandomForest = RandomForestClassifier(max_features=13, n_estimators=80, verbose=2)
    RandomForest.fit(X_train, y_train)
    y_pred = RandomForest.predict(X_test)
    y_pred_proba = RandomForest.predict_proba(X_test)[:,1]
    return y_pred, y_pred_proba