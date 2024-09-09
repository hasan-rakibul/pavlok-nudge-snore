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
    plt.savefig(os.path.join(config.logging_dir, "roc_curve.pdf"), format="pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    config_file = "./config/config.yaml"
    config = OmegaConf.load(config_file)

    y_true = np.load(config.analysis.ground_truth_file)
    y_pred = np.load(config.analysis.prediction_file)

    draw_roc_curve(config, y_true, y_pred)