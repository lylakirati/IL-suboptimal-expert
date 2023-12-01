import torch 
import torch.nn as nn
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader
# import evaluate 
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

class nn_agent(nn.Module):
    def __init__(self, n_layer, state_size, action_size):
        super().__init__()
        self.state_size =  state_size
        self.action_size = action_size
        self.n_layer = n_layer
        layers = nn.Sequential(nn.Linear(self.state_size, self.state_size), nn.ReLU())
        for _ in range(self.n_layer):
            layers.append(nn.Linear(self.state_size, self.state_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.state_size, self.action_size))
        self.layers = layers
        
    
    def forward(self, data):
        return self.layers(data)

class nn_dataset(Dataset):
    def __init__(self, data):
        self.X = data["states"]
        self.y = data["actions"]
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return {
            "states": self.X[idx],
            "actions": self.y[idx]
        }
    
class nn_bc_classifier:
    def __init__(self, model, args):
        self.args = args
        self.model = model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = args.lr)
        self.criterion = nn.CrossEntropyLoss()

    
    def train(self, trainloader, valloader, testloader):
        val_metrics_log = {"f1": [], "accuracy": []}
        test_metrics_log = {"f1": [], "accuracy": []}
        for i in range(self.args.n_epochs):
            tbar = tqdm(trainloader, dynamic_ncols=True)
            for batch in tbar:
                self.optimizer.zero_grad()
                logits = self.model(torch.tensor(batch["states"], dtype = torch.float))
                loss = self.criterion(logits, torch.tensor(batch["actions"], dtype = torch.long))
                loss.backward()
                self.optimizer.step()
            val_metrics = self._eval(valloader)
            test_metrics = self._eval(testloader)
            val_metrics_log["f1"].append(val_metrics["f1"])
            val_metrics_log["accuracy"].append(val_metrics["accuracy"])
            test_metrics_log["f1"].append(test_metrics["f1"])
            test_metrics_log["accuracy"].append(test_metrics["accuracy"])
        return val_metrics_log, test_metrics_log
            
        
    def _eval(self, dataloader):
        preds = []
        actions = []
        self.model.eval()
        tbar = tqdm(dataloader, dynamic_ncols=True)
        with torch.no_grad():
            for batch in tbar:
                logits = self.model(torch.tensor(batch["states"], dtype = torch.float))
                pred = torch.argmax(logits, dim = -1)
                preds.extend(pred.tolist())
                # print(pred.tolist())
                # print(batch["actions"].tolist())
                actions.extend(batch["actions"].tolist())
        # print(actions)
        # print(preds)
        f1 = f1_score(y_true= actions, y_pred=preds, average = "macro")
        accuracy = accuracy_score(y_true= actions, y_pred=preds)
        return {
            "f1": f1,
            "accuracy": accuracy
        }
     
    def experiment(self, trainloader, valloader, testloader):
        val_metrics_log, test_metrics_log = self.train(trainloader, valloader, testloader)
        best_f1_index = np.argmax(val_metrics_log["f1"]).item()
        best_acc_index = np.argmax(val_metrics_log["accuracy"]).item()
        test_f1 = test_metrics_log["f1"][best_f1_index]
        test_acc = test_metrics_log["accuracy"][best_acc_index]
        return {
            "f1": test_f1 ,
            "accuracy": test_acc
        }

    

class sklearn_classifier:
    def __init__(self, model, args = None):
        self.args = args
        self.model = model
    
    def train(self, traindata):
        self.mode.fit(traindata["states"], traindata["actions"])


    def experiment(self, traindata, testdata):
        self.train(traindata)
        preds = self.model.predict(testdata["states"])
        f1 = f1_score(y_true = testdata["labels"], y_pred=preds)
        accuracy = accuracy_score(y_true= testdata["labels"], y_pred=preds)
        return {
            "f1": f1,
            "accuracy": accuracy
        }

def run_experiment(args, traindata, testdata, valdata = None, model = None):
    if args.platform == "nn":
        return run_experiment_nn(args, traindata, valdata, testdata)
    if args.platform == "sklearn":
        return run_experiment_sklearn(model, traindata, testdata)


def run_experiment_sklearn(model, traindata, testdata):
    classifier = sklearn_classifier(model)
    result = classifier.experiment(traindata, testdata)
    return result

def run_experiment_nn(args, traindata = None, valdata = None, testdata = None):
    ## load model
    model = nn_agent(n_layer = args.n_layer, state_size = args.state_size , action_size = args.action_size )

    ## load classifier
    classifier = nn_bc_classifier(model, args)

    ## load dataset 
    traindataset = nn_dataset(traindata)
    valdataset = nn_dataset(valdata)
    testdataset = nn_dataset(testdata)

    ## create data loader
    trainloader = DataLoader(traindataset, batch_size = args.bs, shuffle = True)
    valloader = DataLoader(valdataset, batch_size = args.bs, shuffle = True)
    testloader = DataLoader(testdataset, batch_size = args.bs, shuffle = True)

    ## get results
    result = classifier.experiment(trainloader, valloader, testloader)
    return result

















