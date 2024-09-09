import os
import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['ieee', 'tableau-colorblind10'])

def draw_roc_curve(config, y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(config.analysis.truth_prediction_dir, "roc_curve.pdf"), format="pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    config_file = "./config/config.yaml"
    config = OmegaConf.load(config_file)

    y_true_file = os.path.join(config.analysis.truth_prediction_dir, "y_all.npy")
    y_pred_file = os.path.join(config.analysis.truth_prediction_dir, "y_pred_all.npy")

    assert os.path.exists(y_true_file) and os.path.exists(y_pred_file), \
        "Ground truth and prediction files are required for analysis"

    y_true = np.load(y_true_file)
    y_pred = np.load(y_pred_file)

    draw_roc_curve(config, y_true, y_pred)