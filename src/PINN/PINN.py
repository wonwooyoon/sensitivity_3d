import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import joblib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import math

# Define the DNN model
class DNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hidden_neurons, dropout_rate):
        super(DNNModel, self).__init__()
        layers = [nn.Linear(input_dim, hidden_neurons), nn.GELU(), nn.Dropout(dropout_rate)]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.GELU(), nn.Dropout(dropout_rate)])
        layers.append(nn.Linear(hidden_neurons, output_dim))
        layers.append(nn.Softplus())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class CappedCosineAnnealingWarmRestarts(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_max, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_max = T_max
        self.T_mult = T_mult
        self.eta_min = eta_min

        self.T_i = T_0
        self.last_restart = 0

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch - self.last_restart
        if t >= self.T_i:
            self.last_restart = self.last_epoch
            self.T_i = min(self.T_i * self.T_mult, self.T_max)
            t = 0
            print(f"Restarting at epoch {self.last_epoch}, T_i={self.T_i}")
        
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * t / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]

        
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def criterion_1(output, y_pred):
    return ((output - y_pred) ** 2).mean()

def criterion_2(output, y_pred, device):
    output_true = output * (output_scaler_max - output_scaler_min) + output_scaler_min
    return torch.abs((torch.sum(output_true, dim=1) - 241.500084)).mean()

def denormalize(normalized, min_, max_):
    return normalized * (max_ - min_) + min_

def evaluate_and_save(model, alpha, beta, output_dir):
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train)
        val_pred = model(X_val)
        test_pred = model(X_test)

        y_train_cpu = y_train.cpu()
        y_val_cpu = y_val.cpu()
        y_test_cpu = y_test.cpu()
        train_pred_cpu = train_pred.cpu()
        val_pred_cpu = val_pred.cpu()
        test_pred_cpu = test_pred.cpu()

        output_min = scaler.data_min_[5:]
        output_max = scaler.data_max_[5:]

        y_train_orig = denormalize(y_train_cpu.numpy(), output_min, output_max)
        train_pred_orig = denormalize(train_pred_cpu.numpy(), output_min, output_max)
        y_val_orig = denormalize(y_val_cpu.numpy(), output_min, output_max)
        val_pred_orig = denormalize(val_pred_cpu.numpy(), output_min, output_max)
        y_test_orig = denormalize(y_test_cpu.numpy(), output_min, output_max)
        test_pred_orig = denormalize(test_pred_cpu.numpy(), output_min, output_max)

        for i in range(train_pred_orig.shape[1]):
            plt.figure()
            plt.scatter(y_train_orig[:, i], train_pred_orig[:, i], alpha=0.5, label='Train')
            plt.scatter(y_val_orig[:, i], val_pred_orig[:, i], alpha=0.5, label='Validation')
            plt.scatter(y_test_orig[:, i], test_pred_orig[:, i], alpha=0.5, label='Test')
            min_val = min(y_train_orig[:, i].min(), y_val_orig[:, i].min(), y_test_orig[:, i].min())
            max_val = max(y_train_orig[:, i].max(), y_val_orig[:, i].max(), y_test_orig[:, i].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            plt.xlabel('True')
            plt.ylabel('Predicted')
            plt.title(f'Output {i+1}')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'PINN_combined_output_{i+1}.png'))
            plt.close()

        columns = [f'y{i+1}' for i in range(train_pred_orig.shape[1])] + [f'y{i+1}_pred' for i in range(train_pred_orig.shape[1])]
        pd.DataFrame(np.column_stack([y_train_orig, train_pred_orig]), columns=columns).to_csv(os.path.join(output_dir, 'PINN_train_predictions.csv'), index=False)
        pd.DataFrame(np.column_stack([y_val_orig, val_pred_orig]), columns=columns).to_csv(os.path.join(output_dir, 'PINN_val_predictions.csv'), index=False)
        pd.DataFrame(np.column_stack([y_test_orig, test_pred_orig]), columns=columns).to_csv(os.path.join(output_dir, 'PINN_test_predictions.csv'), index=False)

        r_squared = {}
        for i in range(test_pred_orig.shape[1]):
            ss_res = ((y_test_orig[:, i] - test_pred_orig[:, i]) ** 2).sum()
            ss_tot = ((y_test_orig[:, i] - y_test_orig[:, i].mean()) ** 2).sum()
            r_squared[f'y{i+1}'] = 1 - (ss_res / ss_tot)
        pd.DataFrame(r_squared, index=[0]).to_csv(os.path.join(output_dir, 'PINN_r_squared_values.csv'), index=False)

        sigma_res = {f'y{i+1}': 1.96 * (y_test_orig[:, i] - test_pred_orig[:, i]).std() for i in range(test_pred_orig.shape[1])}
        
        pd.DataFrame(sigma_res, index=[0]).to_csv(os.path.join(output_dir, 'PINN_sigma_values.csv'), index=False)

        nrmse = {}
        for i in range(test_pred_orig.shape[1]):
            nrmse[f'y{i+1}'] = np.sqrt(((y_test_orig[:, i] - test_pred_orig[:, i]) ** 2).mean()) / (y_test_orig[:, i].max() - y_test_orig[:, i].min())
        
        pd.DataFrame(nrmse, index=[0]).to_csv(os.path.join(output_dir, 'PINN_NRMSE_values.csv'), index=False)

# Setup
alpha_set = [1.0] * 5
beta_set = [0.0, 1e-2, 1e-1, 1.0, 10.0]
dropout_rate = 0.1
epochs = 100000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = joblib.load('./src/TargetCalculation/output/minmax_scaler.pkl')
output_scaler_min = torch.tensor(scaler.data_min_[5:], dtype=torch.float32).to(device)
output_scaler_max = torch.tensor(scaler.data_max_[5:], dtype=torch.float32).to(device)
data = pd.read_csv('./src/TargetCalculation/output/normalized_inout.csv')
X = data.iloc[:, 0:5].values
y = data.iloc[:, 5:].values
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=45)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.25, random_state=45)
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

# Optuna tuning and training
for alpha, beta in zip(alpha_set, beta_set):
    print(f"Optimizing for alpha={alpha}, beta={beta}")

    def objective(trial):
        l2_reg = trial.suggest_float('l2_reg', 1e-8, 1e-4, log=True)
        num_layers = trial.suggest_int('num_layers', 3, 6)
        hidden_neurons = trial.suggest_categorical('hidden_neurons', [256, 512, 1024])
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        initial_lr = 1e-4

        model = DNNModel(input_dim, output_dim, num_layers, hidden_neurons, dropout_rate).to(device)
        model.apply(weights_init)
        optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=l2_reg)
        scheduler = CappedCosineAnnealingWarmRestarts(optimizer, T_0=10, T_max=5120, T_mult=2, eta_min=1e-8)
        best_val_loss = float('inf')
        
        num_restarts_without_improvement = 0
        T_0 = 10
        T_mult = 2
        current_T_i = T_0
        next_restart_epoch = T_0

        for epoch in range(epochs):
            model.train()
            permutation = torch.randperm(X_train.size(0))
            for i in range(0, X_train.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_X, batch_y = X_train[indices], y_train[indices]
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = alpha * criterion_1(outputs, batch_y) + beta * criterion_2(outputs, batch_y, device)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_predictions = model(X_val)
                val_loss = alpha * criterion_1(val_predictions, y_val) + beta * criterion_2(val_predictions, y_val, device)

            scheduler.step()
            
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                num_restarts_without_improvement = 0
                print(f"Epoch {epoch + 1}, Best Val Loss: {best_val_loss:.4e}")
            
            if epoch + 1 == next_restart_epoch:
                num_restarts_without_improvement += 1
                if num_restarts_without_improvement >= 2:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
                next_restart_epoch += scheduler.T_i

        return best_val_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=40, show_progress_bar=True)
    best_trial = study.best_trial
    best_params = best_trial.params
    print(f"Best params for alpha={alpha}, beta={beta}: {best_params}")

    model = DNNModel(input_dim, output_dim, best_params['num_layers'], best_params['hidden_neurons'], dropout_rate).to(device)
    model.apply(weights_init)
    optimizer = optim.AdamW(model.parameters(), lr= 1e-4, weight_decay=best_params['l2_reg'])
    scheduler = CappedCosineAnnealingWarmRestarts(optimizer, T_0=10, T_max=5120, T_mult=2, eta_min=1e-8)
    best_val_loss = float('inf')
    endured = 0
    output_dir = f'./src/PINN/output/alpha_{alpha}_beta_{beta}_optuna'
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    pareto_log = []

    num_restarts_without_improvement = 0
    T_0 = 10
    T_mult = 2
    current_T_i = T_0
    next_restart_epoch = T_0

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), best_params['batch_size']):
            indices = permutation[i:i + best_params['batch_size']]
            batch_X, batch_y = X_train[indices], y_train[indices]
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = alpha * criterion_1(outputs, batch_y) + beta * criterion_2(outputs, batch_y, device)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val)
            val_loss = alpha * criterion_1(val_predictions, y_val) + beta * criterion_2(val_predictions, y_val, device)
            train_pred = model(X_train)
            test_pred = model(X_test)

            train_c1 = criterion_1(train_pred, y_train).item()
            train_c2 = criterion_2(train_pred, y_train, device).item()
            val_c1 = criterion_1(val_predictions, y_val).item()
            val_c2 = criterion_2(val_predictions, y_val, device).item()
            test_c1 = criterion_1(test_pred, y_test).item()
            test_c2 = criterion_2(test_pred, y_test, device).item()

        scheduler.step()

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            num_restarts_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)

            pareto_log.append({
                'epoch': epoch + 1,
                'train_criterion_1': train_c1,
                'train_criterion_2': train_c2,
                'val_criterion_1': val_c1,
                'val_criterion_2': val_c2,
                'test_criterion_1': test_c1,
                'test_criterion_2': test_c2
            })
        
        if epoch + 1 == next_restart_epoch:
            num_restarts_without_improvement += 1
            if num_restarts_without_improvement >= 2:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            next_restart_epoch += scheduler.T_i
        
    print(f"Best validation loss: {best_val_loss:.4e}")
    pd.DataFrame([best_params]).to_csv(os.path.join(output_dir, 'PINN_hyperparameters.csv'), index=False)
    pd.DataFrame(pareto_log).to_csv(os.path.join(output_dir, 'PINN_pareto_log.csv'), index=False)
    model.load_state_dict(torch.load(best_model_path))
    evaluate_and_save(model, alpha, beta, output_dir)
