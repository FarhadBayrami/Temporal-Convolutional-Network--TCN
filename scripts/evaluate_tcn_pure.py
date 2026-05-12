import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

print(" Training Pure TCN Model for Casalecchio...\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load data
X_train = np.load('results/X_train_pure.npy')
y_train = np.load('results/y_train_pure.npy')
X_val   = np.load('results/X_val_pure.npy')
y_val   = np.load('results/y_val_pure.npy')
X_test  = np.load('results/X_test_pure.npy')
y_test  = np.load('results/y_test_pure.npy')

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val   = torch.tensor(X_val,   dtype=torch.float32)
y_val   = torch.tensor(y_val,   dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_test  = torch.tensor(y_test,  dtype=torch.float32)

print(f"Training samples: {X_train.shape[0]} | Time steps: {X_train.shape[1]}")

scaler_y = joblib.load('models/scaler_y_pure.pkl')

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val),     batch_size=64, shuffle=False)

# ====================== TCN MODEL ======================
class TCN(nn.Module):
    def __init__(self, input_size=13, num_channels=[64, 128, 64], kernel_size=3, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, num_channels[0], kernel_size, padding='same', dilation=1)
        self.conv2 = nn.Conv1d(num_channels[0], num_channels[1], kernel_size, padding='same', dilation=2)
        self.conv3 = nn.Conv1d(num_channels[1], num_channels[2], kernel_size, padding='same', dilation=4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(num_channels[-1], 48)
        self.fc2 = nn.Linear(48, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = x[:, :, -1]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = TCN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# ====================== TRAINING WITH HISTORY ======================
epochs = 80
patience = 10
best_val_loss = float('inf')
patience_counter = 0

train_losses = []
val_losses = []

print("Starting training...\n")

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        output = model(Xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            output = model(Xb)
            val_loss += criterion(output, yb).item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'models/best_tcn_pure.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

print("\n Training Completed!")

# Load best model
model.load_state_dict(torch.load('models/best_tcn_pure.pth'))
model.eval()

# ====================== EVALUATION ======================
def evaluate(X, y, name):
    model.eval()
    with torch.no_grad():
        pred = model(X.to(device)).cpu().numpy()
    true = scaler_y.inverse_transform(y.numpy())
    pred = scaler_y.inverse_transform(pred)
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    print(f"{name:12} → MAE: {mae:6.2f} | RMSE: {rmse:6.2f}")

print("\n" + "="*70)
print("PURE TCN MODEL PERFORMANCE")
print("="*70)
evaluate(X_train, y_train, "Train")
evaluate(X_val, y_val, "Validation")
evaluate(X_test, y_test, "Test")

# ====================== SAVE TO CSV (YOUR REQUESTED FORMAT) ======================
results = pd.DataFrame({
    'DateTime': test_index.strftime('%Y-%m-%d %H:%M'),
    'Observed_QM': y_true.flatten(),
    'TCN_Prediction': y_pred.flatten(),
    'TOPKAPI_Q': Q_test,
    'TCN_Error': y_true.flatten() - y_pred.flatten(),
    'TOPKAPI_Error': y_true.flatten() - Q_test
})

results.to_csv('results/tcn_pure_comparison.csv', 
               sep=';', 
               index=False, 
               float_format='%.4f')

print(f" Results saved to: results/tcn_pure_comparison.csv")

# ====================== ADVANCED TRAINING CHART ======================
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Training Loss', linewidth=2)
plt.plot(val_losses, label='Validation Loss', linewidth=2)
best_epoch = val_losses.index(min(val_losses))
plt.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Early Stop (Epoch {best_epoch+1})')
plt.scatter(best_epoch, val_losses[best_epoch], color='red', s=100, zorder=5)
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
plt.plot(range(len(val_losses)), val_losses, label='Val Loss')
plt.title('Full Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.axis('off')
info_text = f"""Training Summary:
• Final Train Loss : {train_losses[-1]:.6f}
• Final Val Loss   : {val_losses[-1]:.6f}
• Best Val Loss    : {min(val_losses):.6f}
• Stopped at Epoch : {len(train_losses)}
• Time Steps       : 24
• Features         : 13 (Pure)"""
plt.text(0.05, 0.5, info_text, fontsize=11, va='center', bbox=dict(facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig('results/tcn_training_history.png', dpi=200, bbox_inches='tight')
plt.show()

print("\nTraining chart saved to: results/tcn_training_history.png")
torch.save(model.state_dict(), 'models/final_tcn_pure.pth')
print("Model saved!")