# %%
import torch

# ตรวจสอบและใช้งาน GPU ของชิป M4
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"รันโมเดลบน: {device}")

# %%
import numpy as np
from scipy.io import loadmat
import gc
from torch.utils.data import DataLoader, TensorDataset
import torch

ai_model = 'lstm'
snr = 0
scenario = 'O1'
frequency = 60
antennas = 64

path = f'../DeepMIMO/DeepMIMO/DeepMIMO_dataset/SNR{snr}dB_{scenario}_{frequency}_Ant{antennas}/'
d1 = loadmat(path+'channel1.mat')['a']
d2 = loadmat(path+'channel2.mat')['b']
d3 = loadmat(path+'channel3.mat')['c']

# 2. รวมข้อมูล (ยังเป็น Complex อยู่)
data = np.concatenate((d1, d2, d3), axis=2).transpose(2, 0, 1)
del d1, d2, d3

# 3. แยก Real/Imag แล้วค่อยแปลงเป็น float32 (เพื่อไม่ให้ข้อมูลหาย!)
# ตรงนี้จะทำให้ได้ (Samples, 64, 64)
X_combined = np.concatenate((data.real, data.imag), axis=2).astype(np.float32)
X_flattened = np.reshape(X_combined, (data.shape[0], -1))

del data, X_combined
gc.collect()

# 4. Normalization แบบ Feature-wise (แม่นยำกว่า)
# ปรับสเกลแยกตามแต่ละ Antenna/Subcarrier
mean = np.mean(X_flattened, axis=0)
std = np.std(X_flattened, axis=0)
X_flattened = (X_flattened - mean) / (std + 1e-8)

# 5. โหลด Label
y = loadmat(path + 'DLCB_output.mat')['onehot_label'].astype(np.float32)

# 6. สร้าง Sequence
def create_sequences(data, labels, seq_length):
    num_samples = len(data) - seq_length
    X_seq = np.zeros((num_samples, seq_length, data.shape[1]), dtype=np.float32)
    y_seq = np.zeros((num_samples, labels.shape[1]), dtype=np.float32)
    for i in range(num_samples):
        X_seq[i] = data[i : i + seq_length]
        y_seq[i] = labels[i + seq_length - 1]
    return X_seq, y_seq

seq_length = 5
X_lstm, y_lstm = create_sequences(X_flattened, y, seq_length)
del X_flattened, y
gc.collect()

# 7. Sequential Split
split_idx = int(len(X_lstm) * 0.7)
X_train, X_test = X_lstm[:split_idx], X_lstm[split_idx:]
y_train, y_test = y_lstm[:split_idx], y_lstm[split_idx:]

train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=128, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=128, shuffle=False)

print(f"Fixed LSTM Input shape: {X_train.shape}")

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

class BeamPredictionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, n_beams, dropout_rate):
        super(BeamPredictionLSTM, self).__init__()
        # 1. บีบอัดจาก 4096 -> 180 เพื่อลด Noise ก่อนเข้า LSTM
        self.embedding = nn.Linear(input_size, 180)
        
        # 2. LSTM รับข้อมูลที่คลีนขึ้น (ขนาด 180)
        self.lstm = nn.LSTM(180, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        
        self.fc = nn.Linear(hidden_size, n_beams)

    def forward(self, x):
        # x shape: (Batch, Seq, 4096)
        x = torch.relu(self.embedding(x)) # บีบอัดข้อมูลก่อน
        
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # ใช้ข้อมูลจากจังหวะสุดท้ายมาทำนาย
        return out

# %%
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

model = BeamPredictionLSTM(4096, 416, 2, 64, 0.2).to(device)

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


