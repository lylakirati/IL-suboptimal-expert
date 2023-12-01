import torch 
from torch.nn import nn
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader
import evaluate 


class nn_agent(nn.Module):
    def __init__(self, n_layer, state_size, action_size):
        self.state_size =  state_size
        self.action_size = action_size
        self.n_layer = n_layer
        self.layers = nn.sequential()
        for _ in range(self.n_layer):
            self.layers.append(nn.Linear(self.state_size, self.state_size))
            self.layers.append(nn.ReLU())
        
    
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
        self.f1_computer = evaluate.load_

    
    def train(self, trainloader, valloader, testloader):
        val_metrics_log = {"f1": [], "accuracy": []}
        test_metrics_log = {"f1": [], "accuracy": []}
        for i in range(self.args.n_epochs):
            tbar = tqdm(trainloader, dynamic_ncols=True)
            for batch in tbar:
                self.optimizer.zero_grad()
                logits = self.model(batch["states"])
                loss = self.criterion(logits, batch["actions"])
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
        self.model.eval()
        tbar = tqdm(dataloader, dynamic_ncols=True)
        with torch.no_grad():
            for batch in tbar:
                logits = self.model(batch["states"])
                pred = self.argmax(logits, dim = -1)
                preds.extend(pred.tolist())
        f1 = evaluate.load('f1').compute(reference = dataloader["labels"], predictions=preds)
        accuracy = evaluate.load('accuract').compute(reference = dataloader["labels"], predictions=preds)
        return {
            "f1": f1,
            "accuracy": accuracy
        }
     
    def experiment(self, trainloader, valloader, testloader):
        val_metrics_log, test_metrics_log = self.train(self, trainloader, valloader, testloader)
        best_f1_index = torch.argmax(val_metrics_log["f1"]).item()
        best_acc_index = torch.argmax(val_metrics_log["accuracy"]).item()
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
        f1 = evaluate.load('f1').compute(reference = testdata["labels"], predictions=preds)
        accuracy = evaluate.load('accuract').compute(reference = testdata["labels"], predictions=preds)
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
    model = nn_agent(n_layer = args.n_layer, state_size = args.tate_size , action_size = args.action_size )

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

















