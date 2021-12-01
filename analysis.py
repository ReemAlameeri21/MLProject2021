import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch.nn as nn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

class Analysis:
    def conf_mtrx(test_labels, pred_cls, data):
        test_labels = torch.cat(test_labels).to('cpu')  # cat= concatenate
        pred_cls = torch.cat(pred_cls).to('cpu')

        conf_matrx = confusion_matrix(test_labels, pred_cls)

        class_names = data.classes
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)

        # create heatmap
        sns.heatmap(pd.DataFrame(conf_matrx), annot=True, cmap="YlGnBu", fmt='g')
        ax.xaxis.set_label_position("top")
        plt.tight_layout()
        # You can change the position of the title by providing
        # a value for the y parameter
        plt.title('Confusion matrix [0:Atypical, 1:Indeterminate, 2:Negative, 3:Typical]', y=1.2)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')

    def ROC_plot_AUC_score(test_labels, pred_proba, n_classes, data):
        test_labels = torch.cat(test_labels).to('cpu')  # cat= concatenate
        pred_proba = torch.cat(pred_proba, dim=0).to('cpu')

        soft_func = nn.Softmax(dim=1)
        pred_proba = soft_func(pred_proba)

        fpr = {}  # False postive rate x-axis
        tpr = {}  # True positive rate y-axis
        thresh = {}
        auc_scores = []
        one_vs_all_labels = []

        for i in range(n_classes):
            one_vs_all_labels.append((test_labels == i).numpy().astype('int'))

        for i in range(n_classes):
            fpr[i], tpr[i], thresh[i] = roc_curve(one_vs_all_labels[i], pred_proba[:, i], pos_label=1)
            auc_scores.append(roc_auc_score(one_vs_all_labels[i], pred_proba[:, i]))

        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], linestyle='--', label=f'Class {data.classes[i]} vs Rest')

        plt.title('Multiclass ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')

        # Display AUC scores

        for i in range(n_classes):
            print(f'AUC score of Class {data.classes[i]} vs Rest ===>', auc_scores[i])

    def evaluate_metrics(test_labels, pred_cls, data):
        target_names = data.classes

        test_labels = torch.cat(test_labels).to('cpu')  # cat= concatenate
        pred_cls = torch.cat(pred_cls).to('cpu')

        print(classification_report(test_labels, pred_cls, target_names=target_names))
