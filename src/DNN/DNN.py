import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import joblib

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
l2_reg = 1e-5
num_layers = 6
hidden_neurons = 1024
dropout_rate = 0.1
batch_size = 8

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
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=108)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=108)

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
model = DNNModel(input_dim, output_dim, num_layers, hidden_neurons, dropout_rate).to(device)  # Move model to device

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

optimizer = optim.Adam(model.parameters(), weight_decay=l2_reg)  # L2 regularization via weight_decay

# Training loop with validation
epochs = 10
best_val_loss = float('inf')
best_model_path = './best_model.pth'

# Learning rate scheduler
initial_lr = 0.001
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.85, patience=100, min_lr=1e-6)
endured = 0

for param_group in optimizer.param_groups:
    param_group['lr'] = initial_lr

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_X, batch_y = X_train[indices], y_train[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion_1(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = criterion_1(val_predictions, y_val)
    
    # Step the scheduler based on validation loss
    scheduler.step(val_loss)    

    # Save the best model
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        torch.save(model.state_dict(), best_model_path)
        endured = 0
    else:
        endured += 1
        if endured > 500:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}, Learning Rate: {scheduler.optimizer.param_groups[0]["lr"]:.6f}')

# Load the best model and evaluate on test set
model.load_state_dict(torch.load(best_model_path))
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    test_loss = criterion_1(test_predictions, y_test)
    print(f'Best Validation Loss: {best_val_loss:.4f}')
    print(f'Test Loss: {test_loss.item():.4f}')

# Create directory to save plots if it doesn't exist
output_dir = './src/DNN/output/'
os.makedirs(output_dir, exist_ok=True)

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
        plt.savefig(os.path.join(output_dir, f'DNN_combined_output_{i + 1}.png'))
        plt.close()

# Generate predictions and plot combined graph
model.eval()
with torch.no_grad():
    train_predictions = model(X_train).cpu()
    val_predictions = model(X_val).cpu()
    test_predictions = model(X_test).cpu()
    y_train = y_train.cpu()
    y_val = y_val.cpu() 
    y_test = y_test.cpu()
    plot_combined_graph(train_predictions, y_train, val_predictions, y_val, test_predictions, y_test)

    # Save the ground truth and predictions to CSV files
    train_df = pd.DataFrame({
        'y1': y_train[:, 0].numpy(),
        'y2': y_train[:, 1].numpy(),
        'y3': y_train[:, 2].numpy(),
        'y4': y_train[:, 3].numpy(),
        'y5': y_train[:, 4].numpy(),
        'y6': y_train[:, 5].numpy(),
        'y1_pred': train_predictions[:, 0].numpy(),
        'y2_pred': train_predictions[:, 1].numpy(),
        'y3_pred': train_predictions[:, 2].numpy(),
        'y4_pred': train_predictions[:, 3].numpy(),
        'y5_pred': train_predictions[:, 4].numpy(),
        'y6_pred': train_predictions[:, 5].numpy()
    })
    train_df.to_csv(os.path.join(output_dir, 'DNN_train_predictions.csv'), index=False)

    val_df = pd.DataFrame({
        'y1': y_val[:, 0].numpy(),
        'y2': y_val[:, 1].numpy(),
        'y3': y_val[:, 2].numpy(),
        'y4': y_val[:, 3].numpy(),
        'y5': y_val[:, 4].numpy(),
        'y6': y_val[:, 5].numpy(),
        'y1_pred': val_predictions[:, 0].numpy(),
        'y2_pred': val_predictions[:, 1].numpy(),
        'y3_pred': val_predictions[:, 2].numpy(),
        'y4_pred': val_predictions[:, 3].numpy(),
        'y5_pred': val_predictions[:, 4].numpy(),
        'y6_pred': val_predictions[:, 5].numpy()
    })
    val_df.to_csv(os.path.join(output_dir, 'DNN_val_predictions.csv'), index=False)

    test_df = pd.DataFrame({
        'y1': y_test[:, 0].numpy(),
        'y2': y_test[:, 1].numpy(),
        'y3': y_test[:, 2].numpy(),
        'y4': y_test[:, 3].numpy(),
        'y5': y_test[:, 4].numpy(),
        'y6': y_test[:, 5].numpy(),
        'y1_pred': test_predictions[:, 0].numpy(),
        'y2_pred': test_predictions[:, 1].numpy(),
        'y3_pred': test_predictions[:, 2].numpy(),
        'y4_pred': test_predictions[:, 3].numpy(),
        'y5_pred': test_predictions[:, 4].numpy(),
        'y6_pred': test_predictions[:, 5].numpy()
    })
    test_df.to_csv(os.path.join(output_dir, 'DNN_test_predictions.csv'), index=False)

    # Save the criterion_1 and critertion_2 values to a CSV file 
    device = torch.device('cpu')
    criterion_values = pd.DataFrame({
        'train MSE loss': [criterion_1_separate(train_predictions, y_train).mean().numpy()],
        'train y1 MSE loss': [criterion_1_separate(train_predictions, y_train).numpy()[0]],
        'train y2 MSE loss': [criterion_1_separate(train_predictions, y_train).numpy()[1]],
        'train y3 MSE loss': [criterion_1_separate(train_predictions, y_train).numpy()[2]],
        'train y4 MSE loss': [criterion_1_separate(train_predictions, y_train).numpy()[3]],
        'train y5 MSE loss': [criterion_1_separate(train_predictions, y_train).numpy()[4]],
        'train y6 MSE loss': [criterion_1_separate(train_predictions, y_train).numpy()[5]],
        'train MB loss': [criterion_2_separate(train_predictions, y_train, device).numpy()],
        'val MSE loss': [criterion_1_separate(val_predictions, y_val).mean().numpy()],
        'val y1 MSE loss': [criterion_1_separate(val_predictions, y_val).numpy()[0]],
        'val y2 MSE loss': [criterion_1_separate(val_predictions, y_val).numpy()[1]],
        'val y3 MSE loss': [criterion_1_separate(val_predictions, y_val).numpy()[2]],
        'val y4 MSE loss': [criterion_1_separate(val_predictions, y_val).numpy()[3]],
        'val y5 MSE loss': [criterion_1_separate(val_predictions, y_val).numpy()[4]],
        'val y6 MSE loss': [criterion_1_separate(val_predictions, y_val).numpy()[5]],
        'val MB loss': [criterion_2_separate(val_predictions, y_val, device).numpy()],
        'test MSE loss': [criterion_1_separate(test_predictions, y_test).mean().numpy()],
        'test y1 MSE loss': [criterion_1_separate(test_predictions, y_test).numpy()[0]],
        'test y2 MSE loss': [criterion_1_separate(test_predictions, y_test).numpy()[1]],
        'test y3 MSE loss': [criterion_1_separate(test_predictions, y_test).numpy()[2]],
        'test y4 MSE loss': [criterion_1_separate(test_predictions, y_test).numpy()[3]],
        'test y5 MSE loss': [criterion_1_separate(test_predictions, y_test).numpy()[4]],
        'test y6 MSE loss': [criterion_1_separate(test_predictions, y_test).numpy()[5]],
        'test MB loss': [criterion_2_separate(test_predictions, y_test, device).numpy()]
    })

    criterion_values.to_csv(os.path.join(output_dir, 'DNN_criterion_values.csv'), index=False)