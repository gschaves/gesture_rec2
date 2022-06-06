import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def accuracy(idx_user, gestures, predictions):
    pred_labels_user = []
    actual_labels_user = []

    for count_user in range(0, 1):
        pred_labels = []
        actual_labels = []

        for count_pred in range(1, len(gestures)):
            data = np.array(predictions[idx_user]['Prediction'][count_pred], dtype=object)
            n = 1
            chunks = [int(data[x:x + n][0]) for x in range(0, len(data), n)]
            pred_labels = pred_labels + chunks
        #end

        for count_actual in range(1, len(gestures)):
            data = np.ones(len(
                predictions[idx_user]['Prediction'][count_actual])) * (count_actual + 1)
            n = 1
            chunks = [int(data[x:x + n][0]) for x in range(0, len(data), n)]
            actual_labels = actual_labels + chunks
        #end

        pred_labels_user.append(pred_labels)
        actual_labels_user.append(actual_labels)
    #end

    count_subplot = 0
    cm = {}
    disp = {}

    for count_plot in range(0, 1):
        cm[count_plot] = confusion_matrix(actual_labels_user[count_plot],
                                          pred_labels_user[count_plot])
        disp[count_plot] = ConfusionMatrixDisplay(confusion_matrix=cm[count_plot])
    #end

    acc_aux = 0
    for i in range(0, 1):
        acc_aux += np.trace(cm[i])
    #end
    acc_aux /= np.sum(cm[0])

    return acc_aux
