# %%
import torch

# ตรวจสอบและใช้งาน GPU ของชิป M4
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"รันโมเดลบน: {device}")

# %%
import numpy as np
from scipy.io import loadmat

d1 = loadmat('../DeepMIMO/DeepMIMO/DeepMIMO_dataset/SNR0dB_O1_60_Ant64/channel1.mat')['a']
d2 = loadmat('../DeepMIMO/DeepMIMO/DeepMIMO_dataset/SNR0dB_O1_60_Ant64/channel2.mat')['b']
d3 = loadmat('../DeepMIMO/DeepMIMO/DeepMIMO_dataset/SNR0dB_O1_60_Ant64/channel3.mat')['c']

data = np.concatenate((d1, d2, d3), axis=2).transpose(2, 0, 1)

d_r = data.real.reshape(-1, 1, 64, 32)
d_i = data.imag.reshape(-1, 1, 64, 32)

X = np.concatenate((d_r, d_i), axis=1)

# Load Label (One-hot encoding)
y = loadmat('../DeepMIMO/DeepMIMO/DeepMIMO_dataset/SNR0dB_O1_60_Ant64/DLCB_output.mat')['onehot_label']

print(f"Input shape: {X.shape}")
print(f"Label shape: {y.shape}")

# %%
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

train_ds = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
test_ds = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

class BeamPredictionCNN(nn.Module):
    def __init__(self, n_beams):
        super(BeamPredictionCNN, self).__init__()
        
        # ส่วนสกัดฟีเจอร์ (Conv Layers)
        self.cnn = nn.Sequential(
            # Conv Layer 1: รับ 2 channels, ส่งออก 16 channels, kernel 3x3
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.Dropout(0.1),
            
            # Conv Layer 2: รับ 16 channels, ส่งออก 32 channels
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False),
            
            # Max Pooling 4x4 เพื่อลดขนาด
            # จาก (64, 32) จะเหลือ (16, 8)
            nn.MaxPool2d(kernel_size=4),
            nn.Dropout(0.1)
        )
        
        # ส่วนตัดสินใจ (Fully Connected Layers)
        # คำนวณ Input size: 32 channels * 16 * 8 = 4096
        self.linear = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, n_beams) # ปรับให้ตรงกับจำนวนบีม (64)
        )

    def forward(self, x):
        # สกัดฟีเจอร์ด้วย CNN
        x = self.cnn(x)
        
        # Flatten ข้อมูลเพื่อเข้า Linear layer
        x = x.view(x.size(0), -1) 
        
        # ทำนายผล
        x = self.linear(x)
        return x

# %%
model = BeamPredictionCNN(n_beams=64).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

epochs = 100
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        # Use torch.max to find the best index of beam for CrossEntropyLoss
        loss = criterion(outputs, torch.max(labels, 1)[1])
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# %%
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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# เก็บค่าผลลัพธ์ทั้งหมด
all_preds = []
all_actuals = []

model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        _, actual = torch.max(labels, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_actuals.extend(actual.cpu().numpy())

# สร้าง Confusion Matrix
cm = confusion_matrix(all_actuals, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, cmap='Blues')
plt.title('Confusion Matrix: Predicted vs Actual Beams (SNR 0dB)')
plt.xlabel('Predicted Beam Index')
plt.ylabel('Actual Beam Index')
plt.show()

# %%
plt.figure(figsize=(15, 5))
plt.plot(all_actuals[:200], 'g-', label='Actual Beam (Optimal)', alpha=0.6)
plt.plot(all_preds[:200], 'r--', label='Predicted Beam (DNN)', alpha=0.8)
plt.title('Beam Tracking Performance over User Trajectory (SNR 0dB)')
plt.xlabel('User Index (Sequence)')
plt.ylabel('Beam Index')
plt.legend()
plt.grid(True)
plt.show()


