"""
Credit to: https://www.kaggle.com/pankajj/fashion-mnist-with-pytorch-93-accuracy
"""
import torch
from torch.utils.data import DataLoader, random_split
import optuna
from torch import nn, optim
from load_data import train_loader, test_loader
from model import NeuralNetwork

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def objective(trial):
    lr = trial.suggest_discrete_uniform("lr", 0.000001, 0.1, 0.001)
    n_neurons = trial.suggest_int("n_neurons", 3, 120, 10)

    # train_data, val_data = random_split(train_data,[13167,1463]) # 90%, 10% 

    # train_loader = DataLoader(train_data, batch_size=64)
    # val_loader = DataLoader(val_data)

    model = NeuralNetwork(n_classes=3, n_neurons=n_neurons)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for batch_idx, (images, labels) in enumerate(train_loader):
        # print(f'Batch id: {batch_idx}')
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        correct = 0
        total = 0
        for val_images, val_labels in test_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
        
            outputs = model(val_images)
        
            predictions = torch.max(outputs, 1)[1].to(device)
            correct += (predictions == val_labels).sum()
        
            total += len(val_labels)
        
        accuracy = correct / total 
        trial.report(accuracy, batch_idx)

        if trial.should_prune():
            raise optuna.TrialPruned()

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
    
        outputs = model(images)
    
        predictions = torch.max(outputs, 1)[1].to(device)
        correct += (predictions == labels).sum()
    
        total += len(labels)
    
    accuracy = correct / total 
    return accuracy


if __name__ == "__main__":
    
    sampler = optuna.samplers.TPESampler(seed=10)
    study = optuna.create_study(
        direction='maximize', 
        pruner=optuna.pruners.MedianPruner(
                n_startup_trials=3, n_warmup_steps=10, interval_steps=10), 
        sampler = sampler
        )
    
    study.optimize(objective, n_trials=3) 

    fig = optuna.visualization.plot_intermediate_values(study)
    fig.show()
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.show()