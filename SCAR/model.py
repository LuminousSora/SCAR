import torch
from torch.utils.data import Dataset
from torch import nn
from sklearn.kernel_approximation import RBFSampler
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import skew
import SCAR.train as train

class SCAR:
    def __init__(self, 
                 lr: float = 0.001, 
                 lr_milestones: tuple = (), 
                 batch_size: int = 1024, 
                 epochs: int = 100,
                 rbf: bool = False,
                 a: int = 0.2,
                 device: str = 'cuda',
                 idx = None
                 ):
        self.encoder = None
        self.scl = SCL()
        self.scl2 = SCL2()
        self.rbf=rbf
        self.idx=idx
        self.outlier_detector=None
        self.trainer = train.Trainer(optimizer_name='adam',
                            lr=lr,
                            epochs=epochs,
                            a=a,
                            lr_milestones=lr_milestones,
                            batch_size=batch_size,
                            weight_decay=1e-6,
                            device=device,
                            num_workers=0)
        
    def train_SCL_encoder(self, data_train):
        print('Training Now...')
        train_data=data_train['X']
        labels=data_train['y']
        
        datasets=MyDataset(data=train_data,labels=labels)
        self.encoder=Encoder(train_data.shape[1])
        encoded_data, encoded_data_label = self.trainer.trian(dataset=datasets, encoder=self.encoder, scl=self.scl, scl2=self.scl2)
        if self.rbf:
            rbf = RBFSampler(gamma=0.02, n_components=50, random_state=42)
            encoded_data=rbf.fit_transform(encoded_data)
        self.detector=DETECTOR()
        self.detector.fit(X=encoded_data)

    def test_model(self,data_test):
        print('Testing Now...')
        test_data=data_test['X']
        labels_t=data_test['y']

        datasets_test=MyDataset(data=test_data,labels=labels_t)
        encoded_data=self.trainer.test(dataset=datasets_test, encoder=self.encoder)

        if self.rbf:
            rbf = RBFSampler(gamma=0.02, n_components=50, random_state=42)
            encoded_data=rbf.fit_transform(encoded_data)

        self.detector.test(X_test=encoded_data, y_test=labels_t)
            

class MyDataset(Dataset):#pytorch Dataset
    def __init__(self, data: [] = None, labels: [] = None):
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.size=data.shape[0]
        self.data=torch.tensor(data, dtype=torch.float32)#


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_input = self.data[index]
        label = self.labels[index]
        
        return data_input, label, index

#Encoder
class Encoder(nn.Module):
    def __init__(self, data_dim, rep_dim:int=16):
        super(Encoder, self).__init__()
        self.rep_dim = rep_dim
        self.data_dim = data_dim

        en_layers_num = None

        en_layers_num = [self.data_dim, 128, self.rep_dim]
        
        self.encoder = nn.Sequential(nn.Linear(en_layers_num[0], en_layers_num[1]), 
                                     nn.Tanh(), 
                                     nn.Linear(en_layers_num[1],en_layers_num[2]))
                
    def forward(self, x):
        encoded = self.encoder(x)
        return encoded

class SCL(nn.Module):
    def __init__(self, temperature=0.5, base_temperature=0.5):
        super(SCL, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
       
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size).squeeze()

        return loss
    
class SCL2(nn.Module):

    def __init__(self, temperature=0.5, base_temperature=0.5):
        super(SCL2, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(1-labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        mask_neg_pairs = mask.sum(1)
        mask_neg_pairs = torch.where(mask_neg_pairs < 1e-6, 1, mask_neg_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_neg_pairs

        loss = (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size).squeeze()

        return loss

class DETECTOR():
    def __init__(self, contamination=0.1):
        self.contamination=contamination

    def column_ecdf(self, X):
        ecdf = ECDF(X)
        return ecdf(X)

    def fit(self, X):
        self.X_train = X

    def decision_function(self, X):

        if hasattr(self, 'X_train'):
            original_size = X.shape[0]
            X = np.concatenate((self.X_train, X), axis=0)

        self.U_l = pd.DataFrame(-1*np.log(np.apply_along_axis(self.column_ecdf, 0, X)))
        self.U_r = pd.DataFrame(-1*np.log(np.apply_along_axis(self.column_ecdf, 0, -X)))

        skewness = np.sign(np.apply_along_axis(skew, 0, X))

        self.U_skew = self.U_l * -1*np.sign(skewness - 1) + self.U_r * np.sign(skewness + 1)
        self.O = np.maximum(self.U_skew, np.add(self.U_l, self.U_r)/2)
        if hasattr(self, 'X_train'):

            self.decision_scores_ = self.O.sum(axis=1).to_numpy()[-original_size:]
        else:
            self.decision_scores_ = self.O.sum(axis=1).to_numpy()
        self.threshold_ = np.percentile(self.decision_scores_, (1-self.contamination)*100)
        self.labels_ = np.zeros(len(self.decision_scores_))
        for i in range(len(self.decision_scores_)):
            self.labels_[i] = 1 if self.decision_scores_[i] >= self.threshold_ else 0
        return self.decision_scores_
    
    def test(self, X_test, y_test):
        y_test_scores = self.decision_function(X_test)
        aucroc=np.round(roc_auc_score(y_test, y_test_scores), decimals=4)    
        aucpr = np.round(average_precision_score(y_test, y_test_scores), decimals=4)
        
        print(' AUCROC:{aucroc}, AUCPR:{aucpr}'.format(aucroc=aucroc, aucpr=aucpr))
