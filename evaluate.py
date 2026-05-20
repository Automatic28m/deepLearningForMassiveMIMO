import torch
import torch.nn.functional as F
import numpy as np
import warnings
import csv
import os
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score

# ปิดคำเตือนสำหรับคลาสที่ไม่มีใน Test Set เพื่อให้ Log สะอาด
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

def evaluate_performance(model, test_loader, device, criterion, model_name, ds_config, n_beams=64, csv_file="evaluation_results.csv"):

    model.eval()
    val_loss = 0
    all_preds = []
    all_actuals = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # คำนวณ Loss
            loss = criterion(outputs, torch.max(labels, 1)[1])
            val_loss += loss.item()
            
            # เก็บค่า Probability สำหรับ AUC
            probs = F.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs.data, 1)
            _, actual = torch.max(labels.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_actuals.extend(actual.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # คำนวณ Metrics สุดท้าย
    avg_val_loss = val_loss / len(test_loader)
    avg_val_acc = accuracy_score(all_actuals, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_actuals, all_preds, average='weighted')
    
    all_possible_beams = np.arange(n_beams)
    
    try:
        val_auc = roc_auc_score(
            all_actuals, 
            all_probs, 
            multi_class='ovr', 
            average='weighted', 
            labels=all_possible_beams
        )
    except ValueError:
        val_auc = 0.0

    frequency = ds_config.get('frequency', 'N/A')
    snr = ds_config.get('snr', 'N/A')

    results = {
        "model_name": model_name.upper(),
        "frequency": frequency,
        "snr": snr,
        "loss": avg_val_loss,
        "accuracy": avg_val_acc * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "auc": val_auc * 100,
        "all_preds": all_preds,
        "all_actuals": all_actuals,
        "all_probs": all_probs
    }

    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(['Model', 'Frequency (GHz)', 'SNR (dB)', 'Val Loss', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'AUC (%)'])
        
        writer.writerow([
            results['model_name'],
            results['frequency'],
            results['snr'],
            f"{results['loss']:.4f}",
            f"{results['accuracy']:.2f}%",
            f"{results['precision']:.2f}%",
            f"{results['recall']:.2f}%",
            f"{results['f1']:.2f}%",
            f"{results['auc']:.2f}%"
        ])
        
    print(f"[INFO] Successfully logged results for {results['model_name']} (Freq: {frequency}GHz, SNR: {snr}dB) to {csv_file}")
    
    return results