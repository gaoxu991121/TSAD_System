import numpy as np
from sklearn import metrics


def aucRoc(anomaly_scores : np.ndarray = np.array([]), ground_truth : np.ndarray = np.array([])):
    auc = metrics.roc_auc_score(ground_truth, anomaly_scores)
    return auc

def aucPr(anomaly_scores : np.ndarray = np.array([]), ground_truth : np.ndarray = np.array([])):
    precision, recall, thresholds = metrics.precision_recall_curve(ground_truth, anomaly_scores)

    auc_pr = metrics.auc(recall, precision)

    return auc_pr







def countResult(predict_labels : np.ndarray = np.array([]), ground_truth : np.ndarray = np.array([]), log = False)-> tuple:
    tp = 0
    fp = 0
    tn = 0
    fn = 0



    if log:
        print("predict_labels:",predict_labels)
        print("ground_truth",ground_truth)
    for (index,value) in enumerate(ground_truth):
        predict_value = predict_labels[index]
        if predict_value == 0 and value == 0:
            tn += 1
        elif predict_value == 0 and value == 1:
            fn += 1
        elif predict_value > 0 and value == 1:
            tp += predict_value
        elif predict_value > 0 and value == 0:
            fp += predict_value


    return (tp,fp,tn,fn)

def getLabelIndex(labels = np.array([])):
    result = []
    last_value = 0
    index_start = 0


    for (index, value) in enumerate(labels):
        if value > 0 and last_value < 1:
            index_start = index
            index_end = index
            result.append(index_start)
        elif value > 0 and last_value > 0:
            result.append(index_start)
        elif value < 1 and last_value < 1:
            result.append(-1)
            continue
        elif value < 1 and last_value > 0:
            index_start = 0
            index_end = 0
            result.append(-1)

        last_value = value

    return  result

def findSegment(labels = np.array([])):
    index_start = 0
    index_end = 0
    last_value = 0
    result = []

    for (index,value) in enumerate(labels):

        if value > 0 and last_value < 1 :
            index_start = index
            index_end = index

        elif value > 0 and last_value > 0:
            index_end += 1
        elif value < 1 and last_value < 1:
            continue
        elif value < 1 and last_value > 0:
            result.append([index_start,index_end])
            index_start = 0
            index_end = 0

        last_value = value

    if index_end > 0 :
        result.append([index_start,index_end])

    return result
