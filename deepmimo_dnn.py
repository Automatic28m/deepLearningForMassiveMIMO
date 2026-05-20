# %%
import torch

# ตรวจสอบและใช้งาน GPU ของชิป M4
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"รันโมเดลบน: {device}")

# %%
import numpy as np
from scipy.io import loadmat
import torch.nn as nn
import torch.nn.functional as F
import random
import gc
import os

ai_model = 'DNN'
scenario = 'O1'
antennas = 64
epochs = 100
save_path = './best_models/'
os.makedirs(save_path, exist_ok=True)

frequencies = [28, 60, 140]
snr_list = [5, 10, 15, 20]

# ฟังก์ชันล็อค Seed
def apply_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        
    print(f"Global environment locked with seed: {seed}")

# นิยามคลาสโมเดลดั้งเดิม
class BeamPredictionDNN(nn.Module):
    def __init__(self, input_size, hidden_layers, nodes_per_layer, n_beams, dropout_rate):
        super(BeamPredictionDNN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, nodes_per_layer))
        self.layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers (4 layers)
        for _ in range(hidden_layers):
            self.layers.append(nn.Linear(nodes_per_layer, nodes_per_layer))
            self.layers.append(nn.Dropout(dropout_rate))
            
        # Output layer
        self.output_layer = nn.Linear(nodes_per_layer, n_beams)
        
        # Initialize weights (He Normal)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = F.relu(layer(x))
            else:
                x = layer(x) # Dropout
        
        # Final output uses ReLU as in your previous Keras code
        x = F.relu(self.output_layer(x))
        return x

# เริ่มต้นระบบรันลูปออโต้
for frequency in frequencies:
    for snr in snr_list:
        print(f"\n--- Start configuration: Frequency {frequency} GHz | SNR {snr} dB ---")
        
        # โหลดข้อมูลตาม Path ปัจจุบันของลูป
        path = f'../DeepMIMO/DeepMIMO/DeepMIMO_dataset/SNR{snr}dB_{scenario}_{frequency}_Ant{antennas}/'
        d1 = loadmat(path+'channel1.mat')['a']
        d2 = loadmat(path+'channel2.mat')['b']
        d3 = loadmat(path+'channel3.mat')['c']

        data = np.concatenate((d1, d2, d3), axis=2).transpose(2, 0, 1)

        d_r = data.real
        d_i = data.imag
        data_combined = np.concatenate((d_r, d_i), axis=2)

        # Flatten
        X = np.reshape(data_combined, (data_combined.shape[0], -1))

        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X = (X - mean) / (std + 1e-8) # 1e-8 prevents division by zero

        print("Data normalization complete. Features now have Mean ≈ 0 and Std ≈ 1.")
        # Load Label (One-hot encoding)
        y = loadmat(path+'DLCB_output.mat')['onehot_label']

        print(f"Input shape: {X.shape}")
        print(f"Label shape: {y.shape}")

        # %% การแบ่งชุดข้อมูล
        from torch.utils.data import DataLoader, TensorDataset

        # 70% train, 30% test
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

        # %% ล็อคความสุ่มเริ่มต้น
        apply_global_seed(42)

        # %% สร้างโมเดลขึ้นมาใหม่ในทุกลูป
        model = BeamPredictionDNN(input_size=4096, hidden_layers=1, nodes_per_layer=24, n_beams=64, dropout_rate=0.2).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        criterion = nn.CrossEntropyLoss()

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total Trainable Params: {params}')

        # %% ขั้นตอนการเทรน
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
                # เซฟโมเดลแยกชื่อตามรอบความถี่และ SNR ปัจจุบัน
                torch.save(model.state_dict(), save_path+f'best_{ai_model.lower()}_model_snr{snr}_{scenario}_{frequency}ghz_{antennas}ant.pth')
                print(f"--> Saved better model at Epoch {epoch+1} with Loss: {best_loss:.4f}")

        # %% ขั้นตอนการประเมินผลและวาดกราฟ
        import evaluate as ev
        import visualizer as vis

        # โหลดโมเดลตัวที่ดีที่สุดที่เพิ่งเทรนเสร็จขึ้นมาทำนายผล
        model.load_state_dict(torch.load(save_path+f'best_{ai_model.lower()}_model_snr{snr}_{scenario}_{frequency}ghz_{antennas}ant.pth'))
        model.eval()

        ds_config = {
            'snr': snr,
            'scenario': scenario,
            'frequency': frequency,
            'antennas': antennas
        }  

        # รันชุดวัดผลและพลอตกราฟตามฟังก์ชันเดิมของคุณ
        mimo_results = ev.evaluate_performance(model, test_loader, device, criterion, ai_model)

        vis.plot_training_loss(train_losses, mimo_results, ds_config)
        vis.plot_confusion_matrix(mimo_results['all_actuals'], mimo_results['all_preds'], ai_model, ds_config)
        vis.plot_beam_tracking(mimo_results['all_actuals'], mimo_results['all_preds'], ai_model, ds_config)
        
        # สั่งเคลียร์แรมฝั่ง Python และการ์ดจอ M4 หลังจบแต่ละรอบลูปย่อย
        del d1, d2, d3, data, d_r, d_i, data_combined, X, y, X_train, X_test, y_train, y_test
        del train_ds, test_ds, train_loader, test_loader, model, optimizer, scheduler, criterion
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

print("\n=== FINISH ===")