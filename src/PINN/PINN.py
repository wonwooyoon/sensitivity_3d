import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import joblib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import itertools

# Define the DNN model
class DNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hidden_neurons, dropout_rate):
        super(DNNModel, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_neurons))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout_rate))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(hidden_neurons, output_dim))
        layers.append(nn.Softplus())
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# User inputs
l2_reg_set = [1e-5, 1e-4]
num_layers_set = [4, 6, 8]
hidden_neurons_set = [1024, 2048, 4096]
batch_size_set = [16, 32]
initial_lr_set = [1e-4]

alpha_set = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
beta_set = [0.0, 0.1, 0.5, 0.9, 1.0, 1.1]

dropout_rate = 0.1
epochs = 10000

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load scaler and extract min, max values
scaler_path = './src/TargetCalculation/output/minmax_scaler.pkl'
scaler = joblib.load(scaler_path)
scaler_min = scaler.data_min_
scaler_max = scaler.data_max_
output_scaler_min = torch.tensor(scaler.data_min_[5:], dtype=torch.float32).to(device)
output_scaler_max = torch.tensor(scaler.data_max_[5:], dtype=torch.float32).to(device)

# Load the dataset
data_path = './src/TargetCalculation/output/normalized_inout.csv'
data = pd.read_csv(data_path)

# Split input and output variables
X = data.iloc[:, 0:5].values  # First to fifth columns as input
y = data.iloc[:, 5:].values   # Sixth to last columns as output

# Split into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=44)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=44)

# Convert to PyTorch tensors and move to device
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# Initialize the model, loss function, and optimizer
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

def criterion_1(output, y_pred):
    loss = ((output - y_pred) ** 2).mean()
    return loss

def criterion_2(output, y_pred, device):
    output_true = output * (output_scaler_max.to(device) - output_scaler_min.to(device)) + output_scaler_min.to(device)
    loss = torch.abs((torch.sum(output_true, dim=1) - 241.500084)).mean()
    return loss

def criterion_1_separate(output, y_pred):
    loss = ((output - y_pred) ** 2).mean(dim=0)
    return loss

def criterion_2_separate(output, y_pred, device):
    output_true = output * (output_scaler_max.to(device) - output_scaler_min.to(device)) + output_scaler_min.to(device)
    loss = torch.abs((torch.sum(output_true, dim=1) - 241.500084)).mean(dim=0)
    return loss

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

for alpha, beta in zip(alpha_set, beta_set):

    print(f'Training with alpha: {alpha}, beta: {beta}')

    output_dir = f'./src/PINN/output/alpha_{alpha}_beta_{beta}'
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    best_of_best_model_path = os.path.join(output_dir, 'best_of_best_model.pth')  
    best_of_best_val_loss = float('inf')  

    for l2_reg, num_layers, hidden_neurons, batch_size, initial_lr in itertools.product(l2_reg_set, num_layers_set, hidden_neurons_set, batch_size_set, initial_lr_set):

        print(f'Training with L2 regularization: {l2_reg}, Layers: {num_layers}, Neurons: {hidden_neurons}, Batch size: {batch_size}')

        model = DNNModel(input_dim, output_dim, num_layers, hidden_neurons, dropout_rate).to(device)  # Move model to device
        optimizer = optim.AdamW(model.parameters(), weight_decay=l2_reg)  # L2 regularization via weight_decay
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, min_lr=1e-8)
        model.apply(weights_init)
        best_val_loss = float('inf')
        endured = 0
        pareto_log = []

        for param_group in optimizer.param_groups:
            param_group['lr'] = initial_lr

        # Training loop with learning rate adjustment

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
            
            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                val_predictions = model(X_val)
                val_loss = alpha * criterion_1(val_predictions, y_val) + beta * criterion_2(val_predictions, y_val, device)
            
            # Step the scheduler based on validation loss
            scheduler.step(val_loss)
            
            # Save the best model
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                torch.save(model.state_dict(), best_model_path)
                endured = 0

                with torch.no_grad():
                    
                    train_pred = model(X_train)
                    val_pred = model(X_val)
                    test_pred = model(X_test)
                    train_c1 = criterion_1(train_pred, y_train).item()
                    train_c2 = criterion_2(train_pred, y_train, device).item()
                    val_c1 = criterion_1(val_pred, y_val).item()
                    val_c2 = criterion_2(val_pred, y_val, device).item()
                    test_c1 = criterion_1(test_pred, y_test).item()
                    test_c2 = criterion_2(test_pred, y_test, device).item()
                    
                    pareto_log.append({
                        'epoch': epoch + 1,
                        'train_criterion_1': train_c1,
                        'train_criterion_2': train_c2,
                        'val_criterion_1': val_c1,
                        'val_criterion_2': val_c2,
                        'test_criterion_1': test_c1,
                        'test_criterion_2': test_c2
                    })

            else:
                endured += 1
                if endured > 500:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Print loss every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {loss.item():.4e}, Validation Loss: {val_loss.item():.4e}, Learning Rate: {scheduler.optimizer.param_groups[0]["lr"]:.2e}')
            
        if best_val_loss < best_of_best_val_loss:
            
            best_of_best_val_loss = best_val_loss
            torch.save(model.state_dict(), best_of_best_model_path)
            print(f'New best model saved with validation loss: {best_of_best_val_loss:.4e}')

            # save the hyperparameters for the best model
            best_hyperparams = {
                'l2_reg': l2_reg,
                'num_layers': num_layers,
                'hidden_neurons': hidden_neurons,
                'batch_size': batch_size
            }

            final_pareto_log = pd.DataFrame(pareto_log)

    # Save hyperparameters to a .csv file
    hyperparams_df = pd.DataFrame([best_hyperparams])
    hyperparams_df.to_csv(os.path.join(output_dir, 'PINN_hyperparameters.csv'), index=False)
    final_pareto_log.to_csv(os.path.join(output_dir, 'PINN_pareto_log.csv'), index=False)

    # Create a model instance with the best hyperparameters
    model = DNNModel(input_dim, output_dim, best_hyperparams['num_layers'], best_hyperparams['hidden_neurons'], dropout_rate).to(device)

    # Load the best model and evaluate on test set
    model.load_state_dict(torch.load(best_of_best_model_path))
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)
        test_loss = alpha * criterion_1(test_predictions, y_test) + beta * criterion_2(test_predictions, y_test, device)
        print(f'Best Validation Loss: {best_val_loss:.4e}')
        print(f'Test Loss: {test_loss.item():.4e}')

    # Function to plot and save combined graph
    def plot_combined_graph(train_predictions, train_targets, val_predictions, val_targets, test_predictions, test_targets):
        num_outputs = train_targets.shape[1]
        for i in range(num_outputs):
            plt.figure()
            # Plot training data
            plt.scatter(train_targets[:, i].numpy(), train_predictions[:, i].numpy(), alpha=0.5, label='Train')
            # Plot validation data
            plt.scatter(val_targets[:, i].numpy(), val_predictions[:, i].numpy(), alpha=0.5, label='Validation')
            # Plot test data
            plt.scatter(test_targets[:, i].numpy(), test_predictions[:, i].numpy(), alpha=0.5, label='Test')
            # Diagonal line
            plt.plot([train_targets[:, i].cpu().numpy().min(), train_targets[:, i].cpu().numpy().max()],
                    [train_targets[:, i].cpu().numpy().min(), train_targets[:, i].cpu().numpy().max()], 'r--')
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Output {i + 1}')
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'PINN_combined_output_{i + 1}.png'))
            plt.close()

    # Generate predictions and plot combined graph
    model.eval()
    with torch.no_grad():
        # Ensure all tensors are moved to CPU before converting to numpy
        train_predictions_cpu = model(X_train).detach().cpu()
        val_predictions_cpu = model(X_val).detach().cpu()
        test_predictions_cpu = model(X_test).detach().cpu()
        y_train_cpu = y_train.detach().cpu()
        y_val_cpu = y_val.detach().cpu()
        y_test_cpu = y_test.detach().cpu()
        plot_combined_graph(train_predictions_cpu, y_train_cpu, val_predictions_cpu, y_val_cpu, test_predictions_cpu, y_test_cpu)

        # Save the ground truth and predictions to CSV files (denormalized to original scale)
        # Denormalize function
        def denormalize(normalized, min_, max_):
            return normalized * (max_ - min_) + min_

        # Prepare min and max for outputs
        output_min = scaler.data_min_[5:]
        output_max = scaler.data_max_[5:]

        # Denormalize train, val, test targets and predictions
        y_train_orig = denormalize(y_train_cpu.numpy(), output_min, output_max)
        train_pred_orig = denormalize(train_predictions_cpu.numpy(), output_min, output_max)
        y_val_orig = denormalize(y_val_cpu.numpy(), output_min, output_max)
        val_pred_orig = denormalize(val_predictions_cpu.numpy(), output_min, output_max)
        y_test_orig = denormalize(y_test_cpu.numpy(), output_min, output_max)
        test_pred_orig = denormalize(test_predictions_cpu.numpy(), output_min, output_max)

        train_df = pd.DataFrame({
            'y1': y_train_orig[:, 0],
            'y1_pred': train_pred_orig[:, 0],
            'y2': y_train_orig[:, 1],
            'y2_pred': train_pred_orig[:, 1],
            'y3': y_train_orig[:, 2],
            'y3_pred': train_pred_orig[:, 2],
            'y4': y_train_orig[:, 3],
            'y4_pred': train_pred_orig[:, 3],
            'y5': y_train_orig[:, 4],
            'y5_pred': train_pred_orig[:, 4],
            'y6': y_train_orig[:, 5],
            'y6_pred': train_pred_orig[:, 5]
        })
        train_df.to_csv(os.path.join(output_dir, 'PINN_train_predictions.csv'), index=False)

        val_df = pd.DataFrame({
            'y1': y_val_orig[:, 0],
            'y1_pred': val_pred_orig[:, 0],
            'y2': y_val_orig[:, 1],
            'y2_pred': val_pred_orig[:, 1],
            'y3': y_val_orig[:, 2],
            'y3_pred': val_pred_orig[:, 2],
            'y4': y_val_orig[:, 3],
            'y4_pred': val_pred_orig[:, 3],
            'y5': y_val_orig[:, 4],
            'y5_pred': val_pred_orig[:, 4],
            'y6': y_val_orig[:, 5],
            'y6_pred': val_pred_orig[:, 5]
        })
        val_df.to_csv(os.path.join(output_dir, 'PINN_val_predictions.csv'), index=False)

        test_df = pd.DataFrame({
            'y1': y_test_orig[:, 0],
            'y1_pred': test_pred_orig[:, 0],
            'y2': y_test_orig[:, 1],
            'y2_pred': test_pred_orig[:, 1],
            'y3': y_test_orig[:, 2],
            'y3_pred': test_pred_orig[:, 2],
            'y4': y_test_orig[:, 3],
            'y4_pred': test_pred_orig[:, 3],
            'y5': y_test_orig[:, 4],
            'y5_pred': test_pred_orig[:, 4],
            'y6': y_test_orig[:, 5],
            'y6_pred': test_pred_orig[:, 5]
        })
        
        # Save test predictions to CSV file
        test_df.to_csv(os.path.join(output_dir, 'PINN_test_predictions.csv'), index=False)

        sigma_res = {
            'y1': 1.96*(y_test_orig[:, 0] - test_pred_orig[:, 0]).std(),
            'y2': 1.96*(y_test_orig[:, 1] - test_pred_orig[:, 1]).std(),
            'y3': 1.96*(y_test_orig[:, 2] - test_pred_orig[:, 2]).std(),
            'y4': 1.96*(y_test_orig[:, 3] - test_pred_orig[:, 3]).std(),
            'y5': 1.96*(y_test_orig[:, 4] - test_pred_orig[:, 4]).std(),
            'y6': 1.96*(y_test_orig[:, 5] - test_pred_orig[:, 5]).std()
        }
        
        # Save sigma values to CSV file
        sigma_df = pd.DataFrame(sigma_res, index=[0])
        sigma_df.to_csv(os.path.join(output_dir, 'PINN_sigma_values.csv'), index=False)

        # Calculate R-squared values 

        r_squared = {}
        for i in range(6):
            ss_res = ((y_test_orig[:, i] - test_pred_orig[:, i]) ** 2).sum()
            ss_tot = ((y_test_orig[:, i] - y_test_orig[:, i].mean()) ** 2).sum()
            r_squared[f'y{i+1}'] = 1 - (ss_res / ss_tot)
        r_squared_df = pd.DataFrame(r_squared, index=[0])
        r_squared_df.to_csv(os.path.join(output_dir, 'PINN_r_squared_values.csv'), index=False)
        
        # Save the criterion_1 and critertion_2 values to a CSV file 
        device_test = torch.device('cpu')
        criterion_values = pd.DataFrame({
            'train MSE loss': [criterion_1_separate(train_predictions_cpu, y_train_cpu).mean().numpy()],
            'train y1 MSE loss': [criterion_1_separate(train_predictions_cpu, y_train_cpu).numpy()[0]],
            'train y2 MSE loss': [criterion_1_separate(train_predictions_cpu, y_train_cpu).numpy()[1]],
            'train y3 MSE loss': [criterion_1_separate(train_predictions_cpu, y_train_cpu).numpy()[2]],
            'train y4 MSE loss': [criterion_1_separate(train_predictions_cpu, y_train_cpu).numpy()[3]],
            'train y5 MSE loss': [criterion_1_separate(train_predictions_cpu, y_train_cpu).numpy()[4]],
            'train y6 MSE loss': [criterion_1_separate(train_predictions_cpu, y_train_cpu).numpy()[5]],
            'train MB loss': [criterion_2_separate(train_predictions_cpu, y_train_cpu, device_test).numpy()],
            'val MSE loss': [criterion_1_separate(val_predictions_cpu, y_val_cpu).mean().numpy()],
            'val y1 MSE loss': [criterion_1_separate(val_predictions_cpu, y_val_cpu).numpy()[0]],
            'val y2 MSE loss': [criterion_1_separate(val_predictions_cpu, y_val_cpu).numpy()[1]],
            'val y3 MSE loss': [criterion_1_separate(val_predictions_cpu, y_val_cpu).numpy()[2]],
            'val y4 MSE loss': [criterion_1_separate(val_predictions_cpu, y_val_cpu).numpy()[3]],
            'val y5 MSE loss': [criterion_1_separate(val_predictions_cpu, y_val_cpu).numpy()[4]],
            'val y6 MSE loss': [criterion_1_separate(val_predictions_cpu, y_val_cpu).numpy()[5]],
            'val MB loss': [criterion_2_separate(val_predictions_cpu, y_val_cpu, device_test).numpy()],
            'test MSE loss': [criterion_1_separate(test_predictions_cpu, y_test_cpu).mean().numpy()],
            'test y1 MSE loss': [criterion_1_separate(test_predictions_cpu, y_test_cpu).numpy()[0]],
            'test y2 MSE loss': [criterion_1_separate(test_predictions_cpu, y_test_cpu).numpy()[1]],
            'test y3 MSE loss': [criterion_1_separate(test_predictions_cpu, y_test_cpu).numpy()[2]],
            'test y4 MSE loss': [criterion_1_separate(test_predictions_cpu, y_test_cpu).numpy()[3]],
            'test y5 MSE loss': [criterion_1_separate(test_predictions_cpu, y_test_cpu).numpy()[4]],
            'test y6 MSE loss': [criterion_1_separate(test_predictions_cpu, y_test_cpu).numpy()[5]],
            'test MB loss': [criterion_2_separate(test_predictions_cpu, y_test_cpu, device_test).numpy()]
        })

        criterion_values.to_csv(os.path.join(output_dir, 'PINN_criterion_values.csv'), index=False)