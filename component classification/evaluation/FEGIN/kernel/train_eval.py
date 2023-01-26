import time
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold, KFold
from torch_geometric.data import DenseDataLoader as DenseLoader
from tqdm import tqdm
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import pdb
from sklearn.metrics import roc_auc_score,f1_score
from dataloader import DataLoader  # replace with custom dataloader to handle subgraphs
import netlsd
import pickle
import random
import os
from sklearn.utils import shuffle

def cross_validation_with_val_set(dataset,
                                  model,
                                  folds,
                                  epochs,
                                  batch_size,
                                  lr,
                                  lr_decay_factor,
                                  lr_decay_step_size,
                                  weight_decay,
                                  device, 
                                  logger=None):

    final_train_losses, val_losses, accs, durations = [], [], [], []
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, folds))):

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        pbar = tqdm(range(1, epochs + 1), ncols=70)
        cur_val_losses = []
        cur_accs = []
        for epoch in pbar:
            train_loss = train(model, optimizer, train_loader, device)
            cur_val_losses.append(eval_loss(model, val_loader, device))
            cur_accs.append(eval_acc(model, test_loader, device))
            eval_info = {
                'fold': fold,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': cur_val_losses[-1],
                'test_acc': cur_accs[-1],
            }
            log = 'Fold: %d, train_loss: %0.4f, val_loss: %0.4f, test_acc: %0.4f' % (
                fold, eval_info["train_loss"], eval_info["val_loss"], eval_info["test_acc"]
            )
            pbar.set_description(log)

            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']

        val_losses += cur_val_losses
        accs += cur_accs

        loss, argmin = tensor(cur_val_losses).min(dim=0)
        acc = cur_accs[argmin.item()]
        final_train_losses.append(eval_info["train_loss"])
        log = 'Fold: %d, final train_loss: %0.4f, best val_loss: %0.4f, test_acc: %0.4f' % (
            fold, eval_info["train_loss"], loss, acc
        )
        print(log)
        if logger is not None:
            logger(log)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    loss, argmin = loss.min(dim=1)
    acc = acc[torch.arange(folds, dtype=torch.long), argmin]
    #average_train_loss = float(np.mean(final_train_losses))
    #std_train_loss = float(np.std(final_train_losses))

    log = 'Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.format(
        loss.mean().item(),
        acc.mean().item(),
        acc.std().item(),
        duration.mean().item()
    ) #+ ', Avg Train Loss: {:.4f}'.format(average_train_loss)
    print(log)
    if logger is not None:
        logger(log)

    return loss.mean().item(), acc.mean().item(), acc.std().item()


def cross_validation_without_val_set( dataset,
                                      model,
                                      folds,
                                      epochs,
                                      batch_size,
                                      lr,
                                      lr_decay_factor,
                                      lr_decay_step_size,
                                      weight_decay,
                                      device, 
                                      logger=None):

    
    count = 1
    print("cross validation without val set")
    train_dataset, test_dataset = [],[]
    for d in dataset:
        if d.set=='train':
            train_dataset.append(d)
        else:
            test_dataset.append(d)

    t_start = time.perf_counter()
    test_losses_itr, accs_itr, f1_itr = [], [], []
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    for i in range(5):
        print(f'##################ITERATION #{i} ****************************************')
        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
    # pbar = tqdm(range(1, epochs + 1), ncols=70)
        test_losses, accs, durations,f1_list = [], [], [],[]
        for epoch in range(epochs):
            
            train_loss = train(model, optimizer, train_loader, device)
            test_losses.append(eval_loss(model, test_loader, device))
            auc,f1_ = eval_acc(model, test_loader, device)
            accs.append(auc)
            f1_list.append(f1_)
            eval_info = {
                'epoch': epoch,
                'train_loss': train_loss,
                'test_loss': test_losses[-1],
                'test_acc': accs[-1],
                'test_f1':f1_list[-1]
            }
            # print(eval_info)
            log = ' epoch: %d, train_loss: %0.4f, test_loss: %0.4f, test_acc: %0.4f,test_f1: %0.4f' % (epoch,
                eval_info["train_loss"], eval_info["test_loss"], eval_info["test_acc"],eval_info["test_f1"]
            )
            # pbar.set_description(log)
            print(log)
            
            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']
        test_losses_itr.append(np.min(test_losses))
        accs_itr.append(np.max(accs))
        f1_itr.append(np.max(f1_list))
    if logger is not None:
        logger(log)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_end = time.perf_counter()
    durations.append(t_end - t_start)

    #return loss.mean().item(), acc_final.item(), acc[:, -1].std().item()
    return np.mean(test_losses), np.mean(accs_itr), np.std(accs_itr),np.mean(f1_itr), np.std(f1_itr)

def trainFEGIN( dataset,dataset_name,
                model,
                folds,
                epochs,
                batch_size,
                lr,
                lr_decay_factor,
                lr_decay_step_size,
                weight_decay,
                device, 
                logger=None):

    
    count = 1
    print("cross validation without val set")
    
    if os.path.isfile('data/'+dataset_name+'_netlsd_train.pt'):
        train_des = torch.load('data/'+dataset_name+'_netlsd_train.pt')
        test_des = torch.load('data/'+dataset_name+'_netlsd_test.pt')
        train_dataset = [d for d in dataset if d.set =='train']
        test_dataset = [d for d in dataset if d.set =='test']
    else:
        train_dataset,train_des, test_dataset, test_des = [],[],[],[]
        for index, d in enumerate(dataset):
            if d.set=='train':
                train_des = torch.load('data/ltspice_examples_netlsd_train.pt')
                test_des = torch.load('data/ltspice_examples_netlsd_test.pt')
                des = (torch.tensor(netlsd.heat(to_networkx(d, to_undirected = True)))*0.1).float()
                des_d = Data(x = des, edge_index = d.edge_index, y = d.y)
                train_dataset.append(d)
                train_des.append(des_d)
                # if index%50==0:
                #     print(index)

            else:
                des = (torch.tensor(netlsd.heat(to_networkx(d, to_undirected = True)))*0.1).float()
                des_d = Data(x = des, edge_index = d.edge_index, y = d.y)
                test_des.append(des_d)
                test_dataset.append(d)
    
            torch.save(train_des,'data/'+dataset_name+'_netlsd_train.pt')
            torch.save(test_des,'data/'+dataset_name+'_netlsd_test.pt')
            
    
    train_dataset, train_des = shuffle(train_dataset, train_des)

    t_start = time.perf_counter()
    # print(len(train_dataset), len(train_des), len(test_dataset), len(test_des))
    test_losses_itr, accs_itr, f1_itr = [], [], []
    for i in range(10):
        print(f'##################ITERATION #{i} ****************************************')
        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
            train_loader_des = DataLoader(train_des, batch_size, shuffle=False)
            test_loader_des = DataLoader(test_des, batch_size, shuffle=False)
        
            
    # pbar = tqdm(range(1, epochs + 1), ncols=70)
        test_losses, accs, durations,f1_list = [], [], [],[]
        for epoch in range(epochs):
            
            train_loss = train_FEGIN(model, optimizer, train_loader,train_loader_des,device)
            test_losses.append(eval_loss_FEGIN(model, test_loader,test_loader_des, device))
            f1_ = eval_acc_FEGIN(model, test_loader,test_loader_des, device)
            # accs.append(auc)
            f1_list.append(f1_)
            eval_info = {
                'epoch': epoch,
                'train_loss': train_loss,
                'test_loss': test_losses[-1],
                # 'test_acc': accs[-1],
                'test_f1':f1_list[-1]
            }
            # print(eval_info)
            log = ' epoch: %d, train_loss: %0.4f, test_loss: %0.4f, test_f1: %0.4f' % (epoch,
                eval_info["train_loss"], eval_info["test_loss"],eval_info["test_f1"]
            )
            # pbar.set_description(log)
            print(log)
            
            if epoch % lr_decay_step_size == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_decay_factor * param_group['lr']
        test_losses_itr.append(np.min(test_losses))
        # accs_itr.append(np.max(accs))
        f1_itr.append(np.max(f1_list))
    if logger is not None:
        logger(log)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_end = time.perf_counter()
    durations.append(t_end - t_start)

    #return loss.mean().item(), acc_final.item(), acc[:, -1].std().item()
    return np.mean(test_losses), np.mean(f1_itr), np.std(f1_itr)


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y[dataset.indices()]):
        test_indices.append(torch.from_numpy(idx))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices


def k_fold2(dataset, folds):
    kf = KFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, test_idx in kf.split(dataset):
        test_indices.append(torch.from_numpy(test_idx))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader, device):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        # loss = F.nll_loss(out, data.y.view(-1))
        loss = F.cross_entropy(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)

def train_FEGIN(model, optimizer, loader,des_loader, device):
    model.train()

    total_loss = 0
    for data,des in zip(loader,des_loader):
        optimizer.zero_grad()
        data = data.to(device)
        des = des.to(device)
        out = model(data,des)
        # loss = F.nll_loss(out, data.y.view(-1))
        loss = F.cross_entropy(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)



def eval_acc(model, loader, device):
    model.eval()
    AUC,f1_list = [],[]
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        y = torch.nn.functional.one_hot(data.y.view(-1).cpu(),num_classes = 5)
        p = torch.nn.functional.one_hot(pred.cpu(),num_classes = 5)
        # print("y and P:",y, p)
        # print(pred, pred.shape)
        auc = roc_auc_score(y,p ,average='weighted', multi_class='ovr')  
        f1 =  f1_score(data.y.view(-1).cpu().numpy(), pred.cpu().numpy(), average = "weighted")
        AUC.append(auc)
        f1_list.append(f1)

        # correct += pred.eq(data.y.view(-1)).sum().item()
    return np.mean(AUC),np.mean(f1_list)
    # return correct / len(loader.dataset)
def eval_acc_FEGIN(model, loader, des_loader,device):
    model.eval()
    AUC,f1_list = [],[]
    correct = 0
    for data,des in zip(loader,des_loader):
        data = data.to(device)
        des = des.to(device)
        with torch.no_grad():
            pred = model(data,des).max(1)[1]
        y = torch.nn.functional.one_hot(data.y.view(-1).cpu(),num_classes = 5)
        p = torch.nn.functional.one_hot(pred.cpu(),num_classes = 5)
        # print("y and P:",y, p)
        # print(pred, pred.shape)
        # auc = roc_auc_score(y,p ,average='weighted', multi_class='ovr')  
        f1 =  f1_score(data.y.view(-1).cpu().numpy(), pred.cpu().numpy(), average = "weighted")
        # AUC.append(auc)
        f1_list.append(f1)

        # correct += pred.eq(data.y.view(-1)).sum().item()
    return np.mean(f1_list)

def eval_loss(model, loader, device):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        # loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
        loss += F.cross_entropy(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)
def eval_loss_FEGIN(model, loader,des_loader,device):
    model.eval()

    loss = 0
    for data,des in zip(loader,des_loader):
        data = data.to(device)
        des = des.to(device)
        with torch.no_grad():
            out = model(data,des)
        # loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
        loss += F.cross_entropy(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)
