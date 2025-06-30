import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import shap
import torch.nn as nn
import matplotlib.pyplot as plt

model_dir = os.path.join('./src/PINN/output/alpha_1.0_beta_0.001_optuna')
model_path = os.path.join(model_dir, 'best_model.pth')
hyperparam_path = os.path.join(model_dir, 'PINN_hyperparameters.csv')

hyperparams = pd.read_csv(hyperparam_path, index_col=0)
num_layers = int(hyperparams['num_layers'].iloc[0])
hidden_neurons = int(hyperparams['hidden_neurons'].iloc[0])

input_dim = 5
output_dim = 6
dropout_rate = 0.1

class PINN(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hidden_neurons, dropout_rate):
        super(PINN, self).__init__()
        layers = [nn.Linear(input_dim, hidden_neurons), nn.GELU(), nn.Dropout(dropout_rate)]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.GELU(), nn.Dropout(dropout_rate)])
        layers.append(nn.Linear(hidden_neurons, output_dim))
        layers.append(nn.Softplus())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

model = PINN(input_dim, output_dim, num_layers, hidden_neurons, dropout_rate)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

data = pd.read_csv('./src/TargetCalculation/output/normalized_inout.csv')
X = torch.tensor(data.iloc[-30:, 0:5].values, dtype=torch.float32)
y = torch.tensor(data.iloc[-30:, 5:].values, dtype=torch.float32)

with torch.no_grad():
    y_pred_orig = model(X).cpu().numpy()

y_true = y.cpu().numpy()
orig_mse = mean_squared_error(y_true, y_pred_orig)

importances = []
X_np = X.cpu().numpy()
n_features = X_np.shape[1]
n_repeats = 10

importances = []
for out_idx in range(output_dim):
    feature_importances = []
    y_true_single = y_true[:, out_idx]
    orig_mse_single = mean_squared_error(y_true_single, y_pred_orig[:, out_idx])
    for i in range(n_features):
        permuted_mses = []
        for _ in range(n_repeats):
            X_permuted = X_np.copy()
            np.random.shuffle(X_permuted[:, i])
            X_permuted_tensor = torch.tensor(X_permuted, dtype=torch.float32)
            with torch.no_grad():
                y_pred_perm = model(X_permuted_tensor).cpu().numpy()
            perm_mse = mean_squared_error(y_true_single, y_pred_perm[:, out_idx])
            permuted_mses.append(perm_mse)
        mean_perm_mse = np.mean(permuted_mses)
        importance = (mean_perm_mse - orig_mse_single)
        feature_importances.append(importance)
    importances.append(feature_importances)

importance_df = pd.DataFrame(importances, columns=['perm', 'density', 'pressure', 'pyrite', 'mixing'],
                             index=['Aqueous UO2++ in Granite','Aqueous UO2++ in Bentonite','Aqueous UO2++ in Source','Adsorbed UO2++ in Bentonite','Mineralized UO2++ in Source','Effluxed UO2++'])
importance_df.to_csv(os.path.join(model_dir, 'feature_importances.csv'))

# Define a function for SHAP that takes numpy and returns numpy
def model_predict(x_numpy):
    x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
    with torch.no_grad():
        return model(x_tensor).cpu().numpy()

# Use a subset as background for SHAP (e.g., 20 random samples)
background = X_np[np.random.choice(X_np.shape[0], size=min(20, X_np.shape[0]), replace=False)]

shap_values_list = []
for out_idx in range(output_dim):
    # SHAP expects a function that returns only the output of interest
    def single_output_predict(x_numpy):
        return model_predict(x_numpy)[:, out_idx]
    explainer = shap.KernelExplainer(single_output_predict, background)
    shap_values = explainer.shap_values(X_np, nsamples=100)
    shap_values_list.append(shap_values)

# Save SHAP values for each output as CSV
for out_idx, shap_vals in enumerate(shap_values_list):
    df = pd.DataFrame(shap_vals, columns=['perm', 'density', 'pressure', 'pyrite', 'mixing'])
    df.to_csv(os.path.join(model_dir, f'shap_values_output{out_idx}.csv'), index=False)

# shap value visulization
shap.initjs()
for out_idx, shap_vals in enumerate(shap_values_list):
    shap_values_df = pd.DataFrame(shap_vals, columns=['perm', 'density', 'pressure', 'pyrite', 'mixing'])
    shap.summary_plot(shap_values_df.values, X_np, feature_names=['perm', 'density', 'pressure', 'pyrite', 'mixing'], show=False)
    plt.savefig(os.path.join(model_dir, f'shap_summary_output{out_idx}.png'))
    plt.close()



