import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import csv

save_path = './results/'

def _ensure_dir_exists(sub_dir):
    target_dir = os.path.join(save_path, sub_dir)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


def plot_training_loss(train_losses, val_losses, eval_results, dataset_config, eval_datetime):
    """
    สร้างกราฟ Training Loss พร้อมรายละเอียด Metrics และ Dataset Configuration
    """
    _ensure_dir_exists('result')
    epochs = len(train_losses)
    model = eval_results.get('model_name', 'Model')
    
    final_avg_train = np.mean(train_losses[-10:])
    final_avg_val = np.mean(val_losses[-10:])

    plt.figure(figsize=(10, 8))
    
    plt.plot(range(1, epochs + 1), train_losses, color='blue',
             label='Training Loss', linewidth=1.5)
    
    plt.plot(range(1, epochs + 1), val_losses, color='orange',
             label='Validation Loss', linewidth=1.5, alpha=0.9)

    plt.title(f'{model} Training Performance\n[Generated: {eval_datetime}]',
              fontsize=13, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)

    plt.minorticks_on()
    plt.grid(which='major', linestyle='-',
             linewidth='0.5', color='gray', alpha=0.8)
    plt.grid(which='minor', linestyle='-',
             linewidth='0.5', color='lightgray', alpha=0.5)

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))

    if max(train_losses) > 1.0:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    plt.legend()
    plt.subplots_adjust(bottom=0.38)

    config_text = (
        f"Plot date time: {eval_datetime}\n"
        f"--- Model Configuration ---\n"
        f"Epoch Count: {epochs}\n"
        f"Final Stable Train Loss (Last 10 Epochs): {final_avg_train:.4f}\n"
        f"Final Stable Validation Loss (Last 10 Epochs): {final_avg_val:.4f}\n"
        f"--- Evaluation Metrics for {model} ---\n"
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

    plt.figtext(0.15, 0.02, config_text,
                fontsize=9,
                ha="left",
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))

    filename = f"result_{model.lower()}_{dataset_config['snr']}_{dataset_config['scenario']}_{dataset_config['frequency']}ghz_{dataset_config['antennas']}ant.png"
    plt.savefig(os.path.join(save_path, 'result', filename), dpi=300, bbox_inches='tight')
    print(f"Graph saved as: {filename}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name, dataset_config, eval_datetime):
    _ensure_dir_exists('cm')
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=False, cmap='Blues')
    
    # เพิ่มค่า Timestamp ลงบนหัวข้อหลัก
    plt.title(f'Confusion Matrix: {model_name.upper()}\n[Generated: {eval_datetime}]', fontsize=13, fontweight='bold')
    plt.xlabel('Predicted Beam Index')
    plt.ylabel('Actual Beam Index')

    filename = f"cm_{model_name.lower()}_{dataset_config['snr']}_{dataset_config['scenario']}_{dataset_config['frequency']}ghz_{dataset_config['antennas']}ant.png"
    plt.savefig(os.path.join(save_path, 'cm', filename), dpi=300, bbox_inches='tight')
    print(f"Confusion Matrix saved as: {filename}")
    plt.show()


def plot_beam_tracking(y_true, y_pred, model_name, dataset_config, eval_datetime, sample_range=200):
    _ensure_dir_exists('beam_tracking')
    plt.figure(figsize=(15, 5))
    plt.plot(y_true[:sample_range], 'g-',
             label='Actual Beam (Optimal)', alpha=0.6, linewidth=1.5)
    plt.plot(y_pred[:sample_range], 'r--',
             label=f'Predicted Beam ({model_name.upper()})', alpha=0.8)

    plt.title(f'Beam Tracking Performance: {model_name.upper()}\n[Generated: {eval_datetime}]', fontsize=13, fontweight='bold')
    plt.xlabel('User Index (Sequence)')
    plt.ylabel('Beam Index')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    filename = f"beam_tracking_{model_name.lower()}_{dataset_config['snr']}_{dataset_config['scenario']}_{dataset_config['frequency']}ghz_{dataset_config['antennas']}ant.png"
    plt.savefig(os.path.join(save_path, 'beam_tracking', filename), dpi=300, bbox_inches='tight')
    print(f"Beam Tracking plot saved as: {filename}")
    plt.show()


def plot_se_tracking(user_indices, optimal_se, predicted_se, model_name, ds_config, eval_datetime):
    """
    พล็อตกราฟเปรียบเทียบ Spectral Efficiency ตลอดช่วง Sequence ของ User พร้อมเพิ่มระบบลงเวลากำกับบนหัวเรื่อง
    """
    _ensure_dir_exists('se_tracking')
    plt.figure(figsize=(14, 5))

    # พล็อตเส้น 2 เส้น
    plt.plot(user_indices, optimal_se, label='Optimal SE (Actual Beam)',
             color='green', alpha=0.7, linewidth=1.5)
    plt.plot(user_indices, predicted_se,
             label=f'Achievable SE ({model_name.upper()})', color='red', linestyle='--', alpha=0.8, linewidth=1.5)

    plt.fill_between(user_indices, optimal_se, predicted_se, where=(optimal_se > predicted_se),
                     interpolate=True, color='red', alpha=0.2, label='SE Loss (Misprediction)')

    plt.title(
        f'Spectral Efficiency Tracking: {model_name.upper()} (Freq: {ds_config["frequency"]}GHz, SNR: {ds_config["snr"]}dB)\n[Generated: {eval_datetime}]', fontsize=13, fontweight='bold')
    plt.xlabel('User Index (Sequence)')
    plt.ylabel('Spectral Efficiency (bps/Hz)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')

    plt.tight_layout()
    filename = f"se_tracking_{model_name.lower()}_{ds_config['snr']}_{ds_config['scenario']}_{ds_config['frequency']}ghz_{ds_config['antennas']}ant.png"
    plt.savefig(os.path.join(save_path, 'se_tracking', filename), dpi=300, bbox_inches='tight')
    
    plt.savefig(f"tracking_se_{model_name.lower()}_{ds_config['snr']}dB.png", dpi=300, bbox_inches='tight')
    print(f"Spectral Efficiency Tracking plot saved as: {filename}")
    plt.show()


def save_se_summary_to_csv(mimo_results, se_test, model_name, ds_config, eval_datetime):
    csv_dir = _ensure_dir_exists('csv')
    csv_file = os.path.join(csv_dir, "se_summary.csv")
    
    preds_array = np.array(mimo_results['all_preds'])
    actuals_array = np.array(mimo_results['all_actuals'])
    user_indices = np.arange(len(preds_array))

    predicted_se = se_test[user_indices, preds_array]
    optimal_se = se_test[user_indices, actuals_array]

    mean_se = np.mean(predicted_se)
    min_se = np.min(predicted_se)
    max_se = np.max(predicted_se)

    se_ratio = np.where(optimal_se > 0, predicted_se / optimal_se, 1.0)
    users_above_threshold = np.sum(se_ratio >= 0.90) / len(user_indices) * 100

    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "Time stamp", 
                "Model", 
                "Scenario", 
                "SNR (dB)", 
                "Frequency (GHz)",
                "Antenna Count",
                "Mean SE (bps/Hz)", 
                "Min SE (bps/Hz)", 
                "Max SE (bps/Hz)", 
                "Users with SE ≥ 90% of Optimal (%)"
            ])

        writer.writerow([
            eval_datetime,
            model_name.upper(),
            ds_config['scenario'],
            ds_config['snr'],
            ds_config['frequency'],
            ds_config['antennas'],
            f"{mean_se:.1f}",
            f"{min_se:.1f}",
            f"{max_se:.1f}",
            f"{users_above_threshold:.1f}"
        ])
    print(f"[SUCCESS] Spectral Efficiency report compiled and saved to: {csv_file}")