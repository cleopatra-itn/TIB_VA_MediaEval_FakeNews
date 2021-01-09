from torch import nn
from torch.utils.data import DataLoader, Dataset, sampler, WeightedRandomSampler
import torch
from torch.autograd import Variable
import json
import os, random, copy
import numpy as np
import torch.optim as optim
import time
from sklearn import metrics, preprocessing
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
import argparse

parser = argparse.ArgumentParser(description='Train MLP Models for FakeNews Detection')
parser.add_argument('--bs', type=int, default=32,
                    help='16,32,64,128')
parser.add_argument('--optim', type=str, default='adam',
                    help='sgd, adam')
parser.add_argument('--epochs', type=int, default=100,
                    help='15,20,30')
parser.add_argument('--lr', type=str, default='2e-5',
                    help='1e-5, 5e-5')
parser.add_argument('--gamma', type=float, default=0.75)
parser.add_argument('--step', type=int, default=1,
                    help='any number>1')
parser.add_argument('--ltype', type=int, default=0,
                    help='0-3')
parser.add_argument('--norm', type=int, default=1,
                    help='0 | 1')
parser.add_argument('--split', type=int, default=1,
                    help='1-10')
parser.add_argument('--gpu', type=int, default=0,
                    help='0,1,2,3')


args = parser.parse_args() 

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.fc(x)
        return x * y, y


class UniMLP_SeNet(nn.Module):
    def __init__(self, inp_dim, ncls):
        super(UniMLP_SeNet, self).__init__()
        
        self.se = SELayer(inp_dim)
        self.bn1 = nn.BatchNorm1d(inp_dim)
        self.fc2 = nn.Linear(inp_dim, 128, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.cf = nn.Linear(128, ncls)

        self.dp1 = nn.Dropout(0.2)
        self.dp2 = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, y = self.se(x)
        x = self.dp1(self.bn1(x))
        x = self.dp2(self.relu(self.bn2(self.fc2(x))))

        return self.cf(x), y



class UniDataset(Dataset):
    def __init__(self, feats, labels, normalize=1):
        self.feats = feats
        self.labels = np.array(labels).astype(np.int)
        self.normalize = normalize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feat = self.feats[idx]
        label = self.labels[idx]

        if self.normalize:
            feat = preprocessing.normalize(feat.reshape(1,-1), axis=1).flatten()

        return torch.FloatTensor(feat), torch.tensor(label)


def train(model, optimizer, lr_scheduler, num_epochs):

    since = time.time()

    best_model = model
    best_acc = 0.0
    best_val_loss = 100
    best_epoch = 0

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        since2 = time.time()

        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        tot = 0.0
        cnt = 0
        # Iterate over data.
        for inputs, labels in tr_loader:

            inputs,  labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs, _ = model(inputs)
            _, preds = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)

            # backward + optimize
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()
            tot += len(labels)

            if cnt % 50 == 0:
                print('[%d, %5d] loss: %.5f, Acc: %.2f' %
                      (epoch, cnt + 1, loss.item(), (100.0 * running_corrects) / tot))

            cnt = cnt + 1

        if lr_scheduler:
        	lr_scheduler.step()

        train_loss = running_loss / len(tr_loader)
        train_acc = running_corrects * 1.0 / (len(tr_loader.dataset))

        print('Training Loss: {:.6f} Acc: {:.2f}'.format(train_loss, 100.0 * train_acc))

        val_loss, val_acc, val_mcc = evaluate(model, vl_loader)

        print('Epoch: {:d}, Val Loss: {:.4f}, Val Acc: {:.4f}, Val MCC: {:.4f}'.format(epoch, 
                                            val_loss, val_acc, val_mcc))

        # deep copy the model
        if val_loss <= best_val_loss:
            best_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)

    time_elapsed2 = time.time() - since2
    print('Epoch complete in {:.0f}m {:.0f}s'.format(
        time_elapsed2 // 60, time_elapsed2 % 60))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return best_model, best_epoch


def evaluate(model, loader):
    model.eval()
    test_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:

            inputs, labels = inputs.to(device), labels.to(device)

            outputs, _ = model(inputs)

            preds = torch.argmax(outputs.data, 1)
            
            test_loss += criterion(outputs, labels).item()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

        acc = metrics.accuracy_score(all_labels, all_preds)
        mcc = metrics.matthews_corrcoef(all_labels, all_preds)

    return test_loss/len(loader), acc, mcc



batch_size = args.bs
normalize = args.norm
init_lr = float(args.lr)
epochs = args.epochs
optz = args.optim
step = args.step
ltype = args.ltype
split = args.split

dev_loc = 'dataset/dev/data/'
tr_ids = pd.read_csv(dev_loc+'splits/train%d.txt'%(split), header=None).to_numpy().flatten()
vl_ids = pd.read_csv(dev_loc+'splits/val%d.txt'%(split), header=None).to_numpy().flatten()

layers = ['sent_word_sumavg', 'sent_emb_2_last', 'sent_emb_last', 'sent_word_catavg']
layer = layers[ltype]

feat_text  = json.load(open('features/dev_covidbert.json','r'))

lab_df = pd.read_csv(dev_loc+'all_ids.csv', header=None)[1].to_numpy().flatten()
fname_df = pd.read_csv(dev_loc+'all_ids.csv', header=None)[0].to_numpy().flatten()

dim = 4096 if 'catavg' in layer else 1024   

lab_train = lab_df[tr_ids]
lab_val = lab_df[vl_ids]

ft_train = np.array(feat_text[layer])[tr_ids]
ft_val = np.array(feat_text[layer])[vl_ids]

tr_data = UniDataset(ft_train, lab_train, normalize)
vl_data = UniDataset(ft_val, lab_val, normalize)

tr_loader = DataLoader(dataset=tr_data, batch_size=batch_size, num_workers=2, 
                        shuffle=True)
vl_loader = DataLoader(dataset=vl_data, batch_size=16, num_workers=2)


criterion = nn.CrossEntropyLoss().to(device)
model_ft = UniMLP_SeNet(dim, len(np.unique(lab_train)))

print(model_ft)
model_ft.to(device)

if optz == 'sgd':
    optimizer_ft = optim.SGD(model_ft.parameters(), init_lr, momentum=0.9, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step, gamma=args.gamma)
else:
    optimizer_ft = optim.Adam(model_ft.parameters(), init_lr, weight_decay=1e-5)
    scheduler = None


model_ft, best_epoch = train(model_ft, optimizer_ft, scheduler,num_epochs=epochs)

torch.save(model_ft.state_dict(), 'models/mlp_%s_%d.pt'%(layer, split))

vl_loss, vl_acc, vl_mcc = evaluate(model_ft, vl_loader)
print('Best Epoch: %d, Val Acc: %.4f, %.4f, %.4f'%(best_epoch, np.round(vl_loss,4), 
                                np.round(vl_acc,4), np.round(vl_mcc,4)))