from queue import Empty
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report

from sklearn.metrics import *
import matplotlib.pyplot as plt 

def EvaluatePrediction(y_test, y_pred, y_pred_proba=0):
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)
    print(classification_report(y_test, y_pred))

    #logit_roc_auc = roc_auc_score(y_test, y_pred)
    #fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure()
    #plt.plot(fpr, tpr, label='ROC (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
def EvaluateReggression(y_test, y_pred):
    # The mean squared error
    #print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    #print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))
    return (mean_squared_error(y_test, y_pred), mean_absolute_error(y_test, y_pred), max_error(y_test, y_pred), r2_score(y_test, y_pred))
