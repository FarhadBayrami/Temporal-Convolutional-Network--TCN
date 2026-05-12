import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

print(" Preparing Pure Data-Driven Data for Casalecchio (TCN Model)...\n")

# Load raw data
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
df = df.drop(columns=['Deep','DeepSat','mm','Q'], errors='ignore')   # Remove TOPKAPI Q

print(f"Data loaded! Shape: {df.shape}")

# Features (Pure Data-Driven - No TOPKAPI Q)
features = ['Prec', 'Rain', 'Evap', 'Snow', 'Temp', 'Soil', 'SoilSat', 
            'Perco', 'Surf', 'YSnow', 'EnSnow', 'SWE', 'Inf2Surf']

X = df[features].values
y = df['QM'].values.reshape(-1, 1)

print(f"Using {len(features)} features")

# Scaling
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Save scalers
os.makedirs('models', exist_ok=True)
joblib.dump(scaler_X, 'models/scaler_X_pure.pkl')
joblib.dump(scaler_y, 'models/scaler_y_pure.pkl')

# Create sequences
def create_sequences(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 24
X_seq, y_seq = create_sequences(X_scaled, y_scaled, TIME_STEPS)

train_size = int(len(X_seq) * 0.70)
val_size = int(len(X_seq) * 0.15)

np.save('results/X_train_pure.npy', X_seq[:train_size])
np.save('results/y_train_pure.npy', y_seq[:train_size])
np.save('results/X_val_pure.npy', X_seq[train_size:train_size+val_size])
np.save('results/y_val_pure.npy', y_seq[train_size:train_size+val_size])
np.save('results/X_test_pure.npy', X_seq[train_size+val_size:])
np.save('results/y_test_pure.npy', y_seq[train_size+val_size:])

print(f"\n Pure Data-Driven preparation completed!")
print(f"Time steps: {TIME_STEPS}")
print(f"Training samples: {X_seq[:train_size].shape[0]}")