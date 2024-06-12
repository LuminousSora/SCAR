import numpy as np
from math import ceil

def unlabeled_anomaly_generator(data_train, labeled_anomaly_rate):
    np.random.seed(42)
    idx_normal = np.where(data_train['y'] == 0)[0]
    idx_anomaly = np.where(data_train['y'] == 1)[0]

    idx_labeled_anomaly = np.random.choice(idx_anomaly, ceil(labeled_anomaly_rate * len(idx_anomaly)), replace=False)
    idx_unlabeled_anomaly = np.setdiff1d(idx_anomaly, idx_labeled_anomaly)
    idx_unlabeled = np.append(idx_normal, idx_unlabeled_anomaly)

    del idx_anomaly, idx_unlabeled_anomaly

    # the label of unlabeled data is 0, and that of labeled anomalies is 1
    data_train['y'][idx_unlabeled] = 0
    data_train['y'][idx_labeled_anomaly] = 1
    return data_train
