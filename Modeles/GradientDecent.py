from sklearn.linear_model import SGDClassifier

def GradientDescentPredictor(x_train, y_train, x_test):

    sgd_clf = SGDClassifier(max_iter=90000, loss="log", learning_rate="adaptive", eta0=0.001, n_iter_no_change=7000, random_state=42, verbose=2)
    sgd_clf.fit(x_train, y_train)

    y_pred = sgd_clf.predict(x_test)
    y_pred_proba = sgd_clf.predict_proba(x_test)[:,1]

    return y_pred, y_pred_proba