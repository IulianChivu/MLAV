# Moise Andrei 421 TAID

import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader

class DatasetMNIST(Dataset):
    def __init__(self, cale_catre_date, cale_catre_etichete):

        f = open('train-images.idx3-ubyte', 'r', encoding='latin-1')
        g = open('train-labels.idx1-ubyte', 'r', encoding='latin-1')

        byte = f.read(16) 
        byte_label = g.read(8) 

        mnist_data = np.fromfile(f,dtype=np.uint8).reshape(-1,784)
        mnist_data = mnist_data.reshape(-1,1,28,28)
        mnist_labels = np.fromfile(g,dtype=np.uint8)
 
        self.mnist_data = mnist_data.astype(np.float32)
        self.mnist_labels = mnist_labels.astype(np.int64)

    
    def __len__(self):
        return self.mnist_data.shape[0]
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        date = self.mnist_data[idx,:,:,:]
        etichete = self.mnist_labels[idx]
        
        mnist_batch = {'date': date, 'etichete': etichete}
        
        return mnist_batch



import torch.nn as nn

class Retea_CNN(nn.Module):
    
    def __init__(self, nr_clase):
        super(Retea_CNN, self).__init__()
        
        #out = [(in−K+2P) // S]+1
        
        # nr_img_batch x 1 x 28 x 28 (canale, linii, coloane)
        self.conv_1 = nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = [3,3], stride=[1,1], padding=[1,1])
        self.relu_1 = nn.ReLU()
        
        # nr_img_batch x 4 x 28 x 28 (canale, linii, coloane)
        self.conv_2 = nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = [3,3], stride=[1,1], padding=[1,1])
        self.relu_2 = nn.ReLU()
        
        # nr_img_batch x 8 x 28 x 28 (canale, linii, coloane)
        self.pool_1 = nn.MaxPool2d(kernel_size=[3,3], stride=[1,1])
        
        
        self.fc_1 = nn.Linear(in_features = 8*26*26, out_features=512)
        self.relu_3 = nn.ReLU()
        
        self.drop = nn.Dropout(p=0.5)
        
        self.fc_2 = nn.Linear(in_features = 512, out_features=128)
        self.relu_4 = nn.ReLU()
        
        self.out_layer = nn.Linear(128, nr_clase)
        
        
    def forward(self, input_batch):
       
        x = self.conv_1(input_batch)
        x = self.relu_1(x)
        
        x = self.conv_2(x)
        x = self.relu_2(x)
        
        x = self.pool_1(x)
        
        
        x = torch.flatten(x, 1, 3)
        x = self.fc_1(x)
        x = self.relu_3(x)
        
        x = self.drop(x)
        
        x = self.fc_2(x)
        x = self.relu_4(x)
        
        out = self.out_layer(x)
        
        return out

    
cnn = Retea_CNN(10)
loss_function = nn.CrossEntropyLoss(reduction='sum')
optim = torch.optim.SGD(cnn.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer = optim, milestones=[4,8,12,16], gamma=0.45)


mnistTrain = DatasetMNIST('train-images.idx3-ubyte','train-labels.idx1-ubyte')
trainLoader = DataLoader(mnistTrain, batch_size=128, shuffle=True, num_workers=0)

mnistTest = DatasetMNIST('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte')
testLoader = DataLoader(mnistTest, batch_size=128, shuffle=False, num_workers=0)

nr_epoci = 15

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
cnn.to(device)

cnn = cnn.train()
for ep in range(nr_epoci):
    predictii = []
    etichete = []

    for batch in trainLoader:

        batch_data = batch['date'].to(device)
        batch_labels = batch['etichete'].to(device)
        current_predict = cnn.forward(batch_data)
        current_predict = current_predict.cpu()
        batch_labels = batch_labels.cpu()

        loss = loss_function(current_predict, batch_labels)
        
        current_predict = np.argmax(current_predict.detach().numpy(), axis=1)
        predictii = np.concatenate((predictii,current_predict))
        etichete = np.concatenate((etichete,batch_labels.detach().numpy()))
        
        optim.zero_grad()
        loss.backward()
        optim.step()
     
    scheduler.step()
        
    acc = np.sum(predictii==etichete)/len(predictii)
    print( 'Acuratetea la epoca {} este {}%'.format(ep+1,acc*100) )

    
    
cnn = cnn.eval()  
predictii = []
for batch in testLoader:
    batch_data = batch['date'].to(device)
    batch_labels = batch['etichete'].to(device)

    current_predict = cnn.forward(batch_data)
    
    current_predict = current_predict.cpu()
    batch_labels = batch_labels.cpu()
    
    current_predict = np.argmax(current_predict.detach().numpy(),axis=1)
    predictii = np.concatenate((predictii,current_predict))

acc = np.sum(predictii==mnistTest.mnist_labels)/len(predictii)
print( 'Acuratetea la test este {}%'.format(acc*100) )