"""
Title:       Build a training pipeline for cyber attack prediction that helps user fit models, tune
             hyperparameters, evaluate best models and save results.

Author:      ARYA KONER
"""

import warnings

warnings.filterwarnings("ignore")

import shutil
import argparse
import itertools
import logging
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              BaggingClassifier, GradientBoostingClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from cyber_featureEngineering import ask, read_feature_names, create_dirs
from cyber_trainviz import (plot_predicted_scores, plot_precision_recall,
                      plot_auc_roc, plot_feature_importances)

# ----------------------------------------------------------------------------#
INPUT_DIR = "../processed_data/supervised_learning/"
OUTPUT_DIR = "../evaluations/"
LOG_DIR = "../logs/train/logs/"
PREDICTED_PROBS_DIR = "../logs/train/predicted_probas/"
PREDICTIONS_DIR = "../logs/train/predictions/"
VIZ_DIR = "../logs/train/viz/"

# logging
logger = logging.getLogger('cyber_train')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
logger.addHandler(ch)
fh = logging.FileHandler(LOG_DIR + time.strftime("%Y%m%d-%H%M%S") + '.log')
logger.addHandler(fh)

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------#
def load_features(dir_path, labeled_test=False):
    """
    Load pre-processed feature matrices.

    Inputs:
        - labeled_test (bool): whether test data is labelled

    Returns:
        (N*m array) X_train, (n*m array) X_test,
        (N*1 array) y_train, (n*1 array or None) X_train
            where N is number of training observations, n is number of test
            observations. m is number of features.

    """
    X_train = np.load(dir_path + 'X_train_features.npy')
    y_train = np.load(dir_path + 'y_train_features.npy')
    X_test = np.load(dir_path + 'X_test_features.npy')

    if labeled_test:
        y_test = np.load(dir_path + 'y_test_features.npy')
    else:
        y_test = None

    return X_train, X_test, y_train, y_test


def start_clean():
    """
    Wipe all the folders and documents (predicted probabilities, visualizations,
    predictions, and evaluations the modeling pipeline produces and start clean
    as requested.

    """
    to_clean = [OUTPUT_DIR, PREDICTIONS_DIR, PREDICTIONS_DIR, VIZ_DIR,
                PREDICTED_PROBS_DIR]
    for dir_path in to_clean:
        shutil.rmtree(dir_path)
        create_dirs(dir_path)


class MetricsK:
    """
    Constructed with given relative population threshold k to calculate
    metrics at k% of the population.

    """

    def __init__(self, k):
        self.k = k

    def accuracy_at_k(self, y_true, y_pred_proba):
        """
        Calculate accuracy at k% of the population.

        Inputs:
            - y_true (array): true labels
            - y_pred_proba (array): predicted probabilities

        Returns:
            (float) accuracy at k%

        """
        n = len(y_true)
        k_n = int(n * self.k / 100)

        # Get indices of top k% predictions
        top_k_indices = np.argsort(y_pred_proba)[-k_n:]

        # Calculate accuracy for top k%
        y_true_top_k = y_true[top_k_indices]
        accuracy = np.mean(y_true_top_k)

        return accuracy

    def precision_at_k(self, y_true, y_pred_proba):
        """
        Calculate precision at k% of the population.

        Inputs:
            - y_true (array): true labels
            - y_pred_proba (array): predicted probabilities

        Returns:
            (float) precision at k%

        """
        n = len(y_true)
        k_n = int(n * self.k / 100)

        # Get indices of top k% predictions
        top_k_indices = np.argsort(y_pred_proba)[-k_n:]

        # Calculate precision for top k%
        y_true_top_k = y_true[top_k_indices]
        precision = np.sum(y_true_top_k) / len(y_true_top_k) if len(y_true_top_k) > 0 else 0

        return precision

    def recall_at_k(self, y_true, y_pred_proba):
        """
        Calculate recall at k% of the population.

        Inputs:
            - y_true (array): true labels
            - y_pred_proba (array): predicted probabilities

        Returns:
            (float) recall at k%

        """
        n = len(y_true)
        k_n = int(n * self.k / 100)

        # Get indices of top k% predictions
        top_k_indices = np.argsort(y_pred_proba)[-k_n:]

        # Calculate recall for top k%
        y_true_top_k = y_true[top_k_indices]
        total_positive = np.sum(y_true)
        recall = np.sum(y_true_top_k) / total_positive if total_positive > 0 else 0

        return recall

    def f1_at_k(self, y_true, y_pred_proba):
        """
        Calculate F1 score at k% of the population.

        Inputs:
            - y_true (array): true labels
            - y_pred_proba (array): predicted probabilities

        Returns:
            (float) F1 score at k%

        """
        precision = self.precision_at_k(y_true, y_pred_proba)
        recall = self.recall_at_k(y_true, y_pred_proba)

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return f1


def get_model(model_name):
    """
    Get model instance by name.

    Inputs:
        - model_name (string): name of the model

    Returns:
        (sklearn estimator) model instance

    """
    SEED = 42
    
    DEFAULT_ARGS = {"Logistic Regression": {'random_state': SEED},
                    "Decision Tree": {'random_state': SEED},
                    "Random Forest": {'random_state': SEED,
                                      'oob_score': True,
                                      'n_jobs': -1},
                    "Bagging": {'random_state': SEED,
                                'oob_score': True,
                                'n_jobs': -1},
                    "Ada Boosting": {'random_state': SEED},
                    "Gradient Boosting": {'random_state': SEED},
                    "Naive Bayes": {},
                    "KNN": {'n_jobs': -1},
                    "Linear SVM": {'random_state': SEED}
                    }

    if model_name == "Logistic Regression":
        return LogisticRegression(**DEFAULT_ARGS[model_name])
    elif model_name == "Decision Tree":
        return DecisionTreeClassifier(**DEFAULT_ARGS[model_name])
    elif model_name == "Random Forest":
        return RandomForestClassifier(**DEFAULT_ARGS[model_name])
    elif model_name == "Bagging":
        return BaggingClassifier(**DEFAULT_ARGS[model_name])
    elif model_name == "Ada Boosting":
        return AdaBoostClassifier(**DEFAULT_ARGS[model_name])
    elif model_name == "Gradient Boosting":
        return GradientBoostingClassifier(**DEFAULT_ARGS[model_name])
    elif model_name == "Naive Bayes":
        return GaussianNB(**DEFAULT_ARGS[model_name])
    elif model_name == "KNN":
        return KNeighborsClassifier(**DEFAULT_ARGS[model_name])
    elif model_name == "Linear SVM":
        return LinearSVC(**DEFAULT_ARGS[model_name])
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_grid_search_params(model_name):
    """
    Get grid search parameters for a model.

    Inputs:
        - model_name (string): name of the model

    Returns:
        (dict) grid search parameters

    """
    GRID_SEARCH_PARAMS = {
        "Logistic Regression": {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        },

        "Decision Tree": {
            'max_depth': [3, 5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10]
        },

        "Random Forest": {
            'n_estimators': [100, 300, 500, 1000],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5]
        },

        "Bagging": {
            'n_estimators': [100, 300, 500, 1000],
            'max_samples': [0.05, 0.1, 0.3, 0.5, 0.7, 1.0],
            'max_features': [5, 10, 15, 20, 25, 30]
        },

        "Ada Boosting": {
            'n_estimators': [100, 300, 500, 1000],
            'learning_rate': [0.001, 0.01, 0.1, 1, 10],
            'max_depth': [5, 10, 15, 20]
        },

        "Gradient Boosting": {
            'n_estimators': [100, 300, 500, 1000],
            'learning_rate': [0.001, 0.01, 0.1, 1],
            'max_depth': [3, 5, 10, 15]
        },

        "Naive Bayes": {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
        },

        "KNN": {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        },

        "Linear SVM": {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'loss': ['hinge', 'squared_hinge']
        }
    }

    return GRID_SEARCH_PARAMS.get(model_name, {})


def train_model(X_train, y_train, model_name, hyper_params=None):
    """
    Train a model with given hyperparameters.

    Inputs:
        - X_train (array): training features
        - y_train (array): training labels
        - model_name (string): name of the model
        - hyper_params (dict): hyperparameters

    Returns:
        (sklearn estimator) trained model

    """
    model = get_model(model_name)
    
    if hyper_params:
        model.set_params(**hyper_params)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    logger.info(f"Trained {model_name} in {training_time:.2f} seconds")
    
    return model, training_time


def evaluate_model(model, X_test, y_test, model_name, metrics_k):
    """
    Evaluate a model on test data.

    Inputs:
        - model (sklearn estimator): trained model
        - X_test (array): test features
        - y_test (array): test labels
        - model_name (string): name of the model
        - metrics_k (MetricsK): metrics calculator

    Returns:
        (dict) evaluation results

    """
    start_time = time.time()
    
    # Get predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = model.decision_function(X_test)
    
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    test_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Calculate metrics at k%
    accuracy_k = metrics_k.accuracy_at_k(y_test, y_pred_proba)
    precision_k = metrics_k.precision_at_k(y_test, y_pred_proba)
    recall_k = metrics_k.recall_at_k(y_test, y_pred_proba)
    f1_k = metrics_k.f1_at_k(y_test, y_pred_proba)
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'accuracy_k': accuracy_k,
        'precision_k': precision_k,
        'recall_k': recall_k,
        'f1_k': f1_k,
        'test_time': test_time,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred
    }
    
    return results


def save_results(results_list, output_dir, batch_name):
    """
    Save evaluation results to CSV.

    Inputs:
        - results_list (list): list of evaluation results
        - output_dir (string): output directory
        - batch_name (string): batch name

    """
    # Create results DataFrame
    results_df = []
    
    for i, results in enumerate(results_list):
        row = {
            'ModelIndex': i + 1,
            'ModelName': results['model_name'],
            'HyperGridIndex': 1,  # Simplified for now
            'HyperParams': 'default',
            'TrainingTime (s)': results.get('training_time', 0),
            'TestTime (s)': results['test_time'],
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1': results['f1'],
            'Roc_Auc': results['roc_auc'],
            'Accuracy at 10%': results['accuracy_k'],
            'Precision at 10%': results['precision_k'],
            'Recall at 10%': results['recall_k'],
            'F1 at 10%': results['f1_k']
        }
        results_df.append(row)
    
    results_df = pd.DataFrame(results_df)
    
    # Save to CSV
    output_file = f"{output_dir}{batch_name} - Evaluations.csv"
    results_df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")
    
    return results_df


def main():
    """
    Main function for cyber attack model training.
    """
    parser = argparse.ArgumentParser(description='Cyber Attack Model Training')
    parser.add_argument('--start_clean', type=int, default=0, help='Start with clean directories')
    parser.add_argument('--ask', type=int, default=1, help='Ask user for choices')
    parser.add_argument('--verbose', type=int, default=1, help='Verbose output')
    parser.add_argument('--plot', type=int, default=1, help='Generate plots')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.info("Starting cyber attack model training...")
    
    # Create directories
    create_dirs(OUTPUT_DIR)
    create_dirs(LOG_DIR)
    create_dirs(PREDICTED_PROBS_DIR)
    create_dirs(PREDICTIONS_DIR)
    create_dirs(VIZ_DIR)
    
    if args.start_clean:
        start_clean()
        logger.info("Started with clean directories")
    
    # Load data
    X_train, X_test, y_train, y_test = load_features(INPUT_DIR, labeled_test=True)
    
    if args.verbose:
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Training labels: {np.sum(y_train)} positive out of {len(y_train)}")
        logger.info(f"Test labels: {np.sum(y_test)} positive out of {len(y_test)}")
    
    # Define models to train
    model_names = [
        "Logistic Regression",
        "Decision Tree", 
        "Random Forest",
        "Bagging",
        "Ada Boosting",
        "Gradient Boosting",
        "Naive Bayes",
        "KNN",
        "Linear SVM"
    ]
    
    # Initialize metrics calculator
    metrics_k = MetricsK(k=10)
    
    # Train and evaluate models
    results_list = []
    
    for model_name in model_names:
        logger.info(f"Training {model_name}...")
        
        try:
            # Train model
            model, training_time = train_model(X_train, y_train, model_name)
            
            # Evaluate model
            results = evaluate_model(model, X_test, y_test, model_name, metrics_k)
            results['training_time'] = training_time
            
            results_list.append(results)
            
            logger.info(f"{model_name} - Accuracy: {results['accuracy']:.3f}, "
                       f"Precision: {results['precision']:.3f}, "
                       f"Recall: {results['recall']:.3f}")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            continue
    
    # Save results
    batch_name = "Batch 0"
    results_df = save_results(results_list, OUTPUT_DIR, batch_name)
    
    # Find best model (with error handling)
    if len(results_list) > 0 and 'Accuracy' in results_df.columns:
        best_idx = results_df['Accuracy'].idxmax()
        best_model_name = results_df.loc[best_idx, 'ModelName']
        best_accuracy = results_df.loc[best_idx, 'Accuracy']
        
        logger.info(f"Best model: {best_model_name} with accuracy {best_accuracy:.3f}")
    else:
        logger.warning("No models were successfully trained. Check the data and model configurations.")
        logger.info("Available results columns: " + str(results_df.columns.tolist() if len(results_df) > 0 else "None"))
    
    if args.verbose:
        logger.info("Cyber attack model training completed successfully!")


if __name__ == "__main__":
    main()
