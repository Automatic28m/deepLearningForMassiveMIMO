# %%
import torch

# ตรวจสอบและใช้งาน GPU ของชิป M4
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"รันโมเดลบน: {device}")

# %%
import numpy as np
from scipy.io import loadmat

ai_model = 'cnn'
snr = 0
scenario = 'O1'
frequency = 60
antennas = 64

path = f'../DeepMIMO/DeepMIMO/DeepMIMO_dataset/SNR{snr}dB_{scenario}_{frequency}_Ant{antennas}/'
d1 = loadmat(path+'channel1.mat')['a']
d2 = loadmat(path+'channel2.mat')['b']
d3 = loadmat(path+'channel3.mat')['c']

data = np.concatenate((d1, d2, d3), axis=2).transpose(2, 0, 1)

d_r = data.real.reshape(-1, 1, 64, 32)
d_i = data.imag.reshape(-1, 1, 64, 32)

X = np.concatenate((d_r, d_i), axis=1)
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean) / (std + 1e-8) # 1e-8 prevents division by zero

print("Data normalization complete. Features now have Mean ≈ 0 and Std ≈ 1.")

# Load Label (One-hot encoding)
y = loadmat(path+'DLCB_output.mat')['onehot_label']

print(f"Input shape: {X.shape}")
print(f"Label shape: {y.shape}")

# %%
from torch.utils.data import DataLoader, TensorDataset

split_idx = int(len(X) * 0.7)

# Training set: The first 70% of the user path
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Data Split Complete:")
print(f" - Training samples: {len(X_train)} (First 70%)")
print(f" - Testing samples:  {len(X_test)} (Last 30% - Future Path)")

train_ds = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
test_ds = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

# %%
import torch
import torch.nn as nn
import numpy as np
import random

def apply_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        
    print(f"Global environment locked with seed: {seed}")

apply_global_seed(42)

class BeamPredictionCNN(nn.Module):
    def __init__(self, n_beams, dropout_rate=0.2):
        super(BeamPredictionCNN, self).__init__()
        
        # ส่วนสกัดฟีเจอร์ (Conv Layers)
        self.cnn = nn.Sequential(
            # Conv Layer 1: รับ 2 channels, ส่งออก 16 channels, kernel 3x3
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Conv Layer 2: รับ 16 channels, ส่งออก 48 channels
            nn.Conv2d(in_channels=16, out_channels=48, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Dropout(dropout_rate)
        )
        
        # Fully Connected Layers
        self.linear = nn.Sequential(
            nn.Linear(6144, 511),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(511, n_beams)
        )

    def forward(self, x):
        x = self.cnn(x) # สกัดฟีเจอร์ด้วย CNN
        x = x.view(x.size(0), -1) # Flatten ข้อมูลเพื่อเข้า Linear layer
        x = self.linear(x) # ทำนายผล
        return x

# %%
model = BeamPredictionCNN(n_beams=64, dropout_rate=0.2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
criterion = nn.CrossEntropyLoss()

params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total Trainable Params: {params}')

# %%
epochs = 100
best_loss = float('inf')
train_losses = []

for epoch in range(epochs):
    epoch_loss = 0.0
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        # Use torch.max to find the best index of beam for CrossEntropyLoss
        loss = criterion(outputs, torch.max(labels, 1)[1])
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    avg_epoch_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)
        
    scheduler.step(avg_epoch_loss)
        
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_epoch_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        torch.save(model.state_dict(), f'best_{ai_model}_model.pth')
        print(f"--> Saved better model at Epoch {epoch+1} with Loss: {best_loss:.4f}")

        

# %%
model.load_state_dict(torch.load(f'best_{ai_model}_model.pth'))
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        _, actual = torch.max(labels.data, 1)
        total += labels.size(0)
        correct += (predicted == actual).sum().item()
        
print(f'Average Accuracy on SNR 0dB: {100 * correct / total:.2f}%')

# %%
import evaluate as ev
import visualizer as vis

model.load_state_dict(torch.load(f'best_{ai_model}_model.pth'))

ds_config = {
    'snr': snr,
    'scenario': scenario,
    'frequency': frequency,
    'antennas': antennas
}  

# 1. รันการวัดผลและเก็บค่า raw data
mimo_results = ev.evaluate_performance(model, test_loader, device, criterion, ai_model)

# 2. วาดกราฟ Loss (อันเดิม)
vis.plot_training_loss(train_losses, mimo_results, ds_config)

# 3. วาดกราฟ Confusion Matrix (อันใหม่)
# หมายเหตุ: คุณต้องส่ง all_actuals และ all_preds ที่เก็บมาจากการรันเข้าฟังก์ชันนี้
vis.plot_confusion_matrix(mimo_results['all_actuals'], mimo_results['all_preds'], ai_model)

# 4. วาดกราฟ Beam Tracking (อันใหม่)
vis.plot_beam_tracking(mimo_results['all_actuals'], mimo_results['all_preds'], ai_model)


