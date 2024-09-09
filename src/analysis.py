import os
import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['ieee', 'tableau-colorblind10'])

def draw_roc_curve(config, y_true_files, y_pred_files):
    label_names = config.analysis.label_names
    plt.figure(figsize=(4, 3))
    for i, (y_true_file, y_pred_file) in enumerate(zip(y_true_files, y_pred_files)):
        assert os.path.exists(y_true_file) and os.path.exists(y_pred_file), \
            "Ground truth and prediction files not found"
        y_true = np.load(y_true_file)
        y_pred = np.load(y_pred_file)
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label= label_names[i] + ' ROC (area = %0.2f)' % roc_auc)
    
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    save_dir = config.analysis.save_dir
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "roc_curve.pdf"), format="pdf", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    config_file = "./config/config.yaml"
    config = OmegaConf.load(config_file)

    truth_prediction_dirs = config.analysis.truth_prediction_dirs

    y_true_files, y_pred_files = [], []
    for dir in truth_prediction_dirs:
        y_true_files.append(os.path.join(dir, "y_all.npy"))
        y_pred_files.append(os.path.join(dir, "y_pred_all.npy"))
    
    draw_roc_curve(config, y_true_files, y_pred_files)
