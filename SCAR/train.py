import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
import numpy as np 
from tqdm import tqdm


class Trainer:
    def __init__(self, optimizer_name: str = 'adam', 
                 lr: float = 0.001, 
                 epochs: int = 100,
                 a: int = 0.2,
                 lr_milestones: tuple = (), 
                 batch_size: int = 1024, 
                 weight_decay: float = 1e-6, 
                 device: str = 'cuda',
                 num_workers: int = 0):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.epochs = epochs
        self.a=a
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.num_workers = num_workers
    
    def trian(self, dataset, encoder, scl, scl2):

        encoder=encoder.to(self.device)
        scl=scl.to(self.device)
        scl2=scl2.to(self.device)
        ae_parameters = encoder.parameters()

        train_loader=DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, drop_last=False)

        optimizer = optim.Adam(ae_parameters, lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')
        scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        encoder.train()

        for e in tqdm(range(self.epochs)):
            n_batches = 0
            for data in train_loader:
                data_input, labels, index = data

                data_input = data_input.to(self.device)
                labels=labels.to(self.device)

                encoded = encoder(data_input)
                con_loss = scl(encoded,labels)
                con_loss2 = scl2(encoded,labels)
                
                loss = con_loss
                loss=con_loss+self.a*con_loss2

                optimizer.zero_grad()
                loss = torch.mean(loss) 
                loss.backward()
                optimizer.step()
                scheduler1.step()
                n_batches += 1

        with torch.no_grad():
            n_batches = 0
            for data in train_loader:
                data_input, labels, index = data

                data_input = data_input.to(self.device)
                labels=labels.to(self.device)

                encoded= encoder(data_input)
                
                if n_batches == 0:
                    encoded_data = encoded.cpu().data.numpy()
                    encoded_data_label=labels.cpu().data.numpy()
                else:
                    encoded_data = np.concatenate((encoded_data, encoded.cpu().numpy()), axis=0)
                    encoded_data_label = np.concatenate((encoded_data_label, labels.cpu().numpy()), axis=0)
                
                n_batches += 1
        
        return encoded_data, encoded_data_label

    def test(self, dataset, encoder):
        encoder = encoder.to(self.device)

        test_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False,
                                  num_workers=self.num_workers, drop_last=False)
        n_batches = 0
        encoder.eval()
        with torch.no_grad():
            for data in test_loader:
                data_input, labels, index = data

                data_input = data_input.to(self.device)
                labels=labels.to(self.device)

                encoded = encoder(data_input)
 
                if n_batches == 0:
                    encoded_data = encoded.cpu().data.numpy()
                else:
                    encoded_data = np.concatenate((encoded_data, encoded.cpu().numpy()), axis=0)

                n_batches += 1

        return encoded_data
    
