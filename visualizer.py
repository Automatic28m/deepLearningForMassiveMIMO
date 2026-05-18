import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def plot_training_loss(train_losses, eval_results, dataset_config):
    """
    สร้างกราฟ Training Loss พร้อมรายละเอียด Metrics และ Dataset Configuration
    train_losses: list ของ loss จากการ train
    eval_results: dictionary ที่ได้จากฟังก์ชัน evaluate_performance
    dataset_config: dictionary รวมข้อมูล snr, frequency, antennas, scenario
    """
    epochs = len(train_losses)
    ai_model = eval_results.get('model_name', 'Model')
    initial_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    final_avg_loss = np.mean(train_losses[-10:])

    plt.figure(figsize=(10, 8)) 
    plt.plot(range(1, epochs + 1), train_losses, color='blue', label='Training Loss', linewidth=1.5)

    # Graph styling
    plt.title(f'{ai_model} Training Loss over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray', alpha=0.8)
    plt.grid(which='minor', linestyle='-', linewidth='0.5', color='lightgray', alpha=0.5)

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
    
    # ปรับ Y-axis อัตโนมัติให้เหมาะสมกับค่า Loss
    if max(train_losses) > 1.0:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    plt.legend()
    plt.subplots_adjust(bottom=0.38) # เพิ่มพื้นที่ด้านล่างให้พอดีกับกล่องข้อความ

    # ดึงข้อมูลจาก eval_results และ dataset_config
    config_text = (
        f"Plot date time: {initial_datetime}\n"
        f"--- Model Configuration ---\n"
        f"Epoch Count: {epochs}\n"
        f"Final Stable Loss (Last 10 Epochs): {final_avg_loss:.4f}\n"
        f"--- Evaluation Metrics for {ai_model} ---\n"
        f"Val Loss: {eval_results['loss']:.4f}\n"
        f"Val Accuracy: {eval_results['accuracy']:.2f}%\n"
        f"Val Precision: {eval_results['precision']:.2f}%\n"
        f"Val Recall: {eval_results['recall']:.2f}%\n"
        f"Val F1-Score: {eval_results['f1']:.2f}%\n"
        f"Val AUC: {eval_results['auc']:.2f}%\n"
        f"--- Dataset Configuration ---\n"
        f"SNR: {dataset_config['snr']}dB\n"
        f"Frequency: {dataset_config['frequency']}GHz\n"
        f"Antenna Count: {dataset_config['antennas']}\n"
        f"Scenario Name: DeepMIMO {dataset_config['scenario']}\n"
    )

    # วางกล่องข้อความ (ปรับตำแหน่ง y เป็น 0.02 เพื่อให้อยู่ในกรอบรูปพอดี)
    plt.figtext(0.15, 0.02, config_text, 
                fontsize=9, 
                ha="left", 
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))
    
    # filename = f"result_{ai_model.lower()}_{dataset_config['frequency']}ghz.png"
    # plt.savefig(filename, dpi=300, bbox_inches='tight')
    # print(f"Graph saved as: {filename}")
    plt.show()
    
def plot_confusion_matrix(y_true, y_pred, model_name):
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name.upper()}', fontsize=14)
    plt.xlabel('Predicted Beam Index')
    plt.ylabel('Actual Beam Index')
    
    # filename = f"cm_{model_name.lower()}.png"
    # plt.savefig(filename, dpi=300, bbox_inches='tight')
    # print(f"Confusion Matrix saved as: {filename}")
    plt.show()

def plot_beam_tracking(y_true, y_pred, model_name, sample_range=200):
    plt.figure(figsize=(15, 5))
    plt.plot(y_true[:sample_range], 'g-', label='Actual Beam (Optimal)', alpha=0.6, linewidth=1.5)
    plt.plot(y_pred[:sample_range], 'r--', label=f'Predicted Beam ({model_name.upper()})', alpha=0.8)
    
    plt.title(f'Beam Tracking Performance: {model_name.upper()}', fontsize=14)
    plt.xlabel('User Index (Sequence)')
    plt.ylabel('Beam Index')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # filename = f"tracking_{model_name.lower()}.png"
    # plt.savefig(filename, dpi=300, bbox_inches='tight')
    # print(f"Beam Tracking plot saved as: {filename}")
    plt.show()
