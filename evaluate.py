import torch
import torch.nn.functional as F
import numpy as np
import warnings
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score


# ปิดคำเตือนสำหรับคลาสที่ไม่มีใน Test Set เพื่อให้ Log สะอาด
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

def evaluate_performance(model, test_loader, device, criterion, model_name, n_beams=64):

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

    # เก็บผลลัพธ์ในรูปแบบ Dictionary เพื่อนำไปใช้ต่อได้ง่าย (เช่น ทำกราฟ)
    results = {
        "model_name": model_name.upper(),
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

    # พิมพ์สรุปผล
    # print(f"\n--- Evaluation Metrics for {results['model_name']} ---")
    # print(f"Val Loss:      {results['loss']:.4f}")
    # print(f"Val Accuracy:  {results['accuracy']:.2f}%")
    # print(f"Val Precision: {results['precision']:.2f}%")
    # print(f"Val Recall:    {results['recall']:.2f}%")
    # print(f"Val F1-Score:  {results['f1']:.2f}%")
    # print(f"Val AUC:       {results['auc']:.2f}%")
    
    return results