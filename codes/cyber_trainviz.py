"""
Title:       Visualization functions for cyber attack prediction results.

Author:      ARYA KONER
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os


def plot_predicted_scores(y_true, y_pred_proba, model_name, save_path):
    """
    Plot predicted scores distribution.

    Inputs:
        - y_true (array): true labels
        - y_pred_proba (array): predicted probabilities
        - model_name (string): name of the model
        - save_path (string): path to save the plot

    """
    plt.figure(figsize=(10, 6))
    
    # Plot distribution for each class
    plt.hist(y_pred_proba[y_true == 0], bins=50, alpha=0.7, label='Normal Traffic', color='blue')
    plt.hist(y_pred_proba[y_true == 1], bins=50, alpha=0.7, label='Attack Traffic', color='red')
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title(f'Predicted Scores Distribution - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_precision_recall(y_true, y_pred_proba, model_name, save_path):
    """
    Plot precision-recall curve.

    Inputs:
        - y_true (array): true labels
        - y_pred_proba (array): predicted probabilities
        - model_name (string): name of the model
        - save_path (string): path to save the plot

    """
    from sklearn.metrics import precision_recall_curve
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=model_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_auc_roc(y_true, y_pred_proba, model_name, save_path):
    """
    Plot ROC curve.

    Inputs:
        - y_true (array): true labels
        - y_pred_proba (array): predicted probabilities
        - model_name (string): name of the model
        - save_path (string): path to save the plot

    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importances(model, feature_names, model_name, save_path):
    """
    Plot feature importances.

    Inputs:
        - model (sklearn estimator): trained model
        - feature_names (list): list of feature names
        - model_name (string): name of the model
        - save_path (string): path to save the plot

    """
    # Check if model has feature_importances_
    if not hasattr(model, 'feature_importances_'):
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot top 20 features
    top_n = min(20, len(feature_names))
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Feature Importances - {model_name}')
    plt.bar(range(top_n), importances[indices[:top_n]])
    plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """
    Plot confusion matrix.

    Inputs:
        - y_true (array): true labels
        - y_pred (array): predicted labels
        - model_name (string): name of the model
        - save_path (string): path to save the plot

    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Attack'], 
                yticklabels=['Normal', 'Attack'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_comparison(results_df, save_path):
    """
    Plot model comparison.

    Inputs:
        - results_df (DataFrame): results DataFrame
        - save_path (string): path to save the plot

    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'Roc_Auc']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        if i < 5:  # Only use first 5 subplots
            axes[i].bar(results_df['ModelName'], results_df[metric])
            axes[i].set_title(f'{metric} Comparison')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
    
    # Remove the 6th subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_cyber_visualizations(results_list, output_dir, batch_name):
    """
    Create all cyber attack visualizations.

    Inputs:
        - results_list (list): list of evaluation results
        - output_dir (string): output directory
        - batch_name (string): batch name

    """
    # Create visualization directory
    viz_dir = f"{output_dir}{batch_name}/"
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    # Create subdirectories for each metric
    metrics = ['Accuracy at 10%', 'Precision at 10%', 'Recall at 10%', 'F1 at 10%', 'Roc_Auc at 10%']
    for metric in metrics:
        metric_dir = f"{viz_dir}{metric}/"
        if not os.path.exists(metric_dir):
            os.makedirs(metric_dir)
    
    # Create visualizations for each model
    for results in results_list:
        model_name = results['model_name']
        y_true = results.get('y_true', [])
        y_pred_proba = results['y_pred_proba']
        y_pred = results['y_pred']
        
        # Create model directory
        model_dir = f"{viz_dir}Accuracy at 10%/{model_name}/"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Plot predicted scores
        plot_predicted_scores(y_true, y_pred_proba, model_name, 
                            f"{model_dir}predicted_scores.png")
        
        # Plot precision-recall curve
        plot_precision_recall(y_true, y_pred_proba, model_name, 
                            f"{model_dir}precision_recall.png")
        
        # Plot ROC curve
        plot_auc_roc(y_true, y_pred_proba, model_name, 
                    f"{model_dir}roc_curve.png")
        
        # Plot confusion matrix
        plot_confusion_matrix(y_true, y_pred, model_name, 
                            f"{model_dir}confusion_matrix.png")
    
    # Create model comparison plots
    results_df = pd.DataFrame(results_list)
    plot_model_comparison(results_df, f"{viz_dir}model_comparison.png")
    
    print(f"Visualizations saved to {viz_dir}")


if __name__ == "__main__":
    # Test visualization functions
    print("Cyber attack visualization module loaded successfully!")
