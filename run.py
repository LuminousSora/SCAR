import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split
import math
import SCAR.model as model
import SCAR.utils as utils


if __name__ == '__main__':
    random.seed(42)
    print("cuda:",torch.cuda.is_available())
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_train={}
    data_test={}


    #(0: inliers, 1: outliers)
    train_path='./datasets/cardio.npz'  
    data=np.load(train_path)
    data_train['X'], data_test['X']=train_test_split(data['X'], test_size=0.3, random_state=42)
    data_train['y'], data_test['y']=train_test_split(data['y'], test_size=0.3, random_state=42)

    
    print('training data anomaly rate', np.count_nonzero(data_train['y'] == 1)/data_train['y'].shape[0])
    #generate unlabeled samples in training set
    data_train=utils.unlabeled_anomaly_generator(data_train=data_train, labeled_anomaly_rate=0.05)
    print('training data labeled anomaly rate', np.count_nonzero(data_train['y'] == 1)/data_train['y'].shape[0])
    print('data train rows:', data_train['y'].shape[0])
    print('data test rows:', data_test['y'].shape[0])

    #If memory permits, we recommend setting the batch size to 1/10 of the number of rows in the dataset.
    batch_size = math.floor(data_train['y'].shape[0]/7)
    #Set the max batch size according to your memory limit
    if batch_size > 13000:
        batch_size = 13000
    print('batch size:', batch_size)

    run = model.SCAR(lr=0.001, lr_milestones=[], batch_size=batch_size, epochs=100, rbf=True, a=0.2, device=device)
    run.train_SCL_encoder(data_train=data_train)
    run.test_model(data_test=data_test)
    
