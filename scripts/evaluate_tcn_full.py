import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("🚀 Evaluating Pure TCN on FULL Dataset...\n")

# ====================== LOAD RAW DATA ======================
df = pd.read_csv('data/425.sbs.ts', 
                 sep=r'\s+', 
                 skiprows=1, 
                 header=None,
                 names=['YYYY','MM','DD','HH','mm','QM','Q','Rain','Prec','Evap','Snow',
                        'Temp','Etp','Soil','SoilSat','Perco','Surf','YSnow','EnSnow',
                        'SWE','Deep','DeepSat','Inf2Surf'])

df['datetime'] = pd.to_datetime({
    'year': df['YYYY'],
    'month': df['MM'],
    'day': df['DD'],
    'hour': df['HH']
})
df = df.set_index('datetime')
df = df.drop(columns=['Deep','DeepSat','mm'], errors='ignore')

# ====================== LOAD ALL DATA ======================
X_train = np.load('results/X_train_pure.npy')
X_val   = np.load('results/X_val_pure.npy')
X_test  = np.load('results/X_test_pure.npy')

X_full = np.concatenate([X_train, X_val, X_test], axis=0)

y_train = np.load('results/y_train_pure.npy')
y_val   = np.load('results/y_val_pure.npy')
y_test  = np.load('results/y_test_pure.npy')
y_full  = np.concatenate([y_train, y_val, y_test], axis=0)

scaler_y = joblib.load('models/scaler_y_pure.pkl')

# ====================== TCN MODEL ======================
class TCN(torch.nn.Module):
    def __init__(self, input_size=13, num_channels=[64, 128, 64], kernel_size=3, dropout=0.2):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(input_size, num_channels[0], kernel_size, padding='same', dilation=1)
        self.conv2 = torch.nn.Conv1d(num_channels[0], num_channels[1], kernel_size, padding='same', dilation=2)
        self.conv3 = torch.nn.Conv1d(num_channels[1], num_channels[2], kernel_size, padding='same', dilation=4)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(num_channels[-1], 48)
        self.fc2 = torch.nn.Linear(48, 1)

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

model = TCN()
model.load_state_dict(torch.load('models/best_tcn_pure.pth', weights_only=True))
model.eval()

# ====================== PREDICTIONS ON ALL DATA ======================
device = torch.device('cpu')
X_full_tensor = torch.tensor(X_full, dtype=torch.float32).to(device)

with torch.no_grad():
    pred_scaled = model(X_full_tensor).cpu().numpy()

y_true = scaler_y.inverse_transform(y_full)
y_pred = scaler_y.inverse_transform(pred_scaled)

# Get TOPKAPI
test_index = df.index[-len(y_true):]   # Last part matches the sequences
Q_full = df.loc[test_index, 'Q'].values

# ====================== SAVE TO CSV ======================
results = pd.DataFrame({
    'DateTime': test_index.strftime('%Y-%m-%d %H:%M'),
    'Observed_QM': y_true.flatten(),
    'TCN_Prediction': y_pred.flatten(),
    'TOPKAPI_Q': Q_full,
    'TCN_Error': y_true.flatten() - y_pred.flatten(),
    'TOPKAPI_Error': y_true.flatten() - Q_full
})

results.to_csv('results/tcn_pure_full_comparison.csv', 
               sep=';', 
               index=False, 
               float_format='%.4f')

print(f"✅ Full predictions saved to: results/tcn_pure_full_comparison.csv")

# ====================== METRICS ======================
def print_metrics(name, true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    print(f"{name:20} → MAE: {mae:6.2f} | RMSE: {rmse:6.2f}")

print("\n" + "="*75)
print("PURE TCN FULL DATASET PERFORMANCE")
print("="*75)
print_metrics("Pure TCN", y_true, y_pred)
print_metrics("TOPKAPI", y_true, Q_full)

# ====================== DOWNSAMPLED PLOTS ======================
step = 100000

plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(y_true[::step], label='Observed QM', linewidth=1.8)
plt.plot(y_pred[::step], label='Pure TCN', linewidth=1.6)
plt.plot(Q_full[::step], label='TOPKAPI', linewidth=1.6, alpha=0.9)
plt.title('Full Dataset - Pure TCN vs TOPKAPI (Downsampled)')
plt.ylabel('Discharge (m³/s)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
start = -2000
plt.plot(y_true[start:], label='Observed QM', linewidth=2)
plt.plot(y_pred[start:], label='Pure TCN', linewidth=1.8)
plt.plot(Q_full[start:], label='TOPKAPI', linewidth=1.8, alpha=0.85)
plt.title('High Flow Period - Last 2000 Hours')
plt.ylabel('Discharge (m³/s)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
plt.plot(y_true - y_pred, label='Pure TCN Error', alpha=0.8)
plt.plot(y_true - Q_full, label='TOPKAPI Error', alpha=0.8)
plt.title('Error Comparison')
plt.ylabel('Error (m³/s)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/tcn_pure_full_plot.png', dpi=200, bbox_inches='tight')
plt.show()

print("\n✅ Full evaluation completed!")