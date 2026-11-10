"""MLP membership-inference attack (Section 7, Appendix E).

An MLP is trained on the 11-dimensional profile of mean reconstruction errors (one feature per lambda, from
``rf_membership.mia.reconstruction``) to predict member / non-member. Depth, width and epochs are selected with Optuna on a
validation split of the MLP training set; the final MLP is evaluated once on held-out samples (AUC, TPR@1%/5% FPR,
confusion matrices).

    python -m rf_membership.mia.mlp_attack --recon-dir outputs/run1/recon --out-dir outputs/run1/mlp
"""

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import optuna
from optuna.trial import Trial
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc



# =============================================================================
# Configuration
# =============================================================================


BALANCE_TEST = False
DENOISING_STEP = 1
TARGET_FPR = [0.01, 0.05]

# Paramètres Optuna
N_TRIALS = 25
N_MODELS = 1  # Fixe pour l'optimisation (plus rapide)


# =============================================================================
# Modèle
# =============================================================================
class SingleModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], num_classes=2, dropout_rate=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# =============================================================================
# Deep Ensemble Classifier
# =============================================================================
class DeepEnsembleClassifier:
    
    def __init__(self, n_models=5, hidden_dims=[128, 64], dropout_rate=0.3):
        self.n_models = n_models
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.models = []
        self.scaler = StandardScaler()
        self.feature_names = []
        self.classes_ = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def extract_features_from_csv(self, csv_path):
        """Features = mean reconstruction error at each lambda (one column per lambda)."""
        df = pd.read_csv(csv_path)
        self.feature_names = [f"mean_t{c}" for c in df.columns]
        return df.values

    def prepare_data_memorization(self, train_seen_csv, train_unseen_csv, 
                                  test_seen_csv, test_unseen_csv):
        
        X_train_seen = self.extract_features_from_csv(train_seen_csv)
        X_train_unseen = self.extract_features_from_csv(train_unseen_csv)
        X_test_seen = self.extract_features_from_csv(test_seen_csv)
        X_test_unseen = self.extract_features_from_csv(test_unseen_csv)
        
        # 1=seen, 0=unseen
        y_train = np.concatenate([np.ones(len(X_train_seen)), np.zeros(len(X_train_unseen))])
        y_test = np.concatenate([np.ones(len(X_test_seen)), np.zeros(len(X_test_unseen))])
        
        X_train = np.vstack([X_train_seen, X_train_unseen])
        X_test = np.vstack([X_test_seen, X_test_unseen])
        
        train_idx = np.random.permutation(len(X_train))
        X_train, y_train = X_train[train_idx], y_train[train_idx]
        
        test_idx = np.random.permutation(len(X_test))
        X_test, y_test = X_test[test_idx], y_test[test_idx]

        return X_train, y_train, X_test, y_test
    
    def train(self, X_train, y_train, epochs=100, lr=0.001, batch_size=128):
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Poids de classe pour équilibrage
        self.classes_ = np.unique(y_train)
        class_weights = compute_sample_weight('balanced', y_train)
        weight_per_class = np.array([class_weights[y_train == c].mean() 
                                     for c in self.classes_])
        
        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(weight_per_class).to(self.device)
        )
        
        X_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_tensor = torch.LongTensor(y_train).to(self.device)
        
        # Entraîner chaque modèle avec une initialisation différente
        self.models = []
        for i in range(self.n_models):
            print(f"Training Model {i+1}/{self.n_models}")
            
            model = SingleModel(
                input_dim=X_train_scaled.shape[1],
                hidden_dims=self.hidden_dims,
                num_classes=len(self.classes_),
                dropout_rate=self.dropout_rate
            ).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
            
            model.train()
            for epoch in range(epochs):
                # Mélanger les données à chaque epoch
                indices = torch.randperm(len(X_tensor))
                
                total_loss = 0
                n_batches = 0
                
                for start_idx in range(0, len(X_tensor), batch_size):
                    batch_idx = indices[start_idx:start_idx + batch_size]
                    batch_X = X_tensor[batch_idx]
                    batch_y = y_tensor[batch_idx]
                    
                    optimizer.zero_grad()
                    logits = model(batch_X)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    n_batches += 1
                
                if (epoch + 1) % 50 == 0:
                    avg_loss = total_loss / n_batches
                    
                    model.eval()
                    with torch.no_grad():
                        logits = model(X_tensor)
                        pred = logits.argmax(dim=1)
                        acc = (pred == y_tensor).float().mean()
                    model.train()
                    
                    print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Acc: {acc.item():.4f}")
            
            self.models.append(model)

    def predict_proba(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        X_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(X_tensor)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs.cpu().numpy())
        
        predictions = np.array(predictions)  # (n_models, n_samples, n_classes)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        return mean_pred, std_pred
    
    def evaluate(self, X_test, y_test, output_dir, balance_test=True):
        
        if balance_test:
            X_test, y_test = self.balance_dataset(X_test, y_test)
        
        y_pred_proba, y_pred_std = self.predict_proba(X_test)
        
        mean_uncertainty = y_pred_std.mean(axis=0)
        print(f"\nMean prediction uncertainty: {mean_uncertainty}")
        
        optimal_threshold, optimal_ba, roc_auc, tpr_at_fpr = self.plot_roc_and_find_optimal_threshold(
            y_test, y_pred_proba, output_dir, TARGET_FPR
        )

        y_pred_05 = (y_pred_proba[:, 1] > 0.5).astype(int)
        self.plot_confusion_matrix(y_test, y_pred_05, 0.5, output_dir, 
                                   'confusion_matrix_threshold_0.5.png')
        self.plot_confusion_by_confidence(y_test, y_pred_proba, 0.5, output_dir,
                                         'confusion_by_confidence_threshold_0.5.png')
        
        y_pred_opt = (y_pred_proba[:, 1] > optimal_threshold).astype(int)
        self.plot_confusion_matrix(y_test, y_pred_opt, optimal_threshold, output_dir,
                                   f'confusion_matrix_threshold_{optimal_threshold:.3f}.png')
        self.plot_confusion_by_confidence(y_test, y_pred_proba, optimal_threshold, output_dir,
                                         f'confusion_by_confidence_threshold_{optimal_threshold:.3f}.png')
        
        self.plot_uncertainty_distribution(y_test, y_pred_std, optimal_threshold, y_pred_proba, output_dir)

        threshold_key = f'threshold_{optimal_threshold:.3f}'
        
        results = {
            'ensemble_config': {
                'n_models': self.n_models,
                'hidden_dims': self.hidden_dims,
            },
            'mean_uncertainty': {
                'class_0': float(mean_uncertainty[0]),
                'class_1': float(mean_uncertainty[1])
            },
            'roc_auc': float(roc_auc),
            'optimal_threshold': float(optimal_threshold),
            'optimal_balanced_accuracy': float(optimal_ba),
            'threshold_0.5': {
                'accuracy': float(accuracy_score(y_test, y_pred_05)),
                'report': classification_report(y_test, y_pred_05, output_dict=True, zero_division=0)
            },
            threshold_key: {
                'accuracy': float(accuracy_score(y_test, y_pred_opt)),
                'report': classification_report(y_test, y_pred_opt, output_dict=True, zero_division=0)
            }, 
            'tpr @ fpr': tpr_at_fpr
        }
        
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

    def balance_dataset(self, X, y):
        class_counts = Counter(y)
        min_size = min(class_counts.values())
        
        indices = []
        for cls in class_counts.keys():
            cls_indices = np.where(y == cls)[0]
            selected = np.random.choice(cls_indices, min_size, replace=False)
            indices.extend(selected)
        
        indices = np.array(indices)
        np.random.shuffle(indices)
        
        return X[indices], y[indices]

    def plot_roc_and_find_optimal_threshold(self, y_test, y_pred_proba, output_dir, target_fprs=[0.01, 0.05]):
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        balanced_accuracies = (tpr + (1 - fpr)) / 2
        best_idx = np.argmax(balanced_accuracies)
        best_ba = balanced_accuracies[best_idx]
        best_threshold = thresholds[best_idx]
        
        tpr_at_fpr = {}
        for target_fpr in target_fprs:
            idx = np.argmin(np.abs(fpr - target_fpr))
            tpr_at_fpr[target_fpr] = {
                'tpr': float(tpr[idx]),
                'actual_fpr': float(fpr[idx]),
                'threshold': float(thresholds[idx])
            }
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        optimal_fpr = fpr[best_idx]
        optimal_tpr = tpr[best_idx]
        plt.scatter(optimal_fpr, optimal_tpr, color='red', s=100, zorder=5,
                label=f'Optimal (thresh={best_threshold:.3f})')
        
        for target_fpr in target_fprs:
            idx = np.argmin(np.abs(fpr - target_fpr))
            plt.scatter(fpr[idx], tpr[idx], color='green', s=80, zorder=5, marker='x')
            plt.text(fpr[idx] + 0.02, tpr[idx], f'FPR={target_fpr:.0%}\nTPR={tpr[idx]:.1%}', 
                    fontsize=8, va='center')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        title = f'ROC Curve - Deep Ensemble ({self.n_models} models)'
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
        plt.close()
        
        return best_threshold, best_ba, roc_auc, tpr_at_fpr
    
    def plot_confusion_matrix(self, y_test, y_pred, threshold, output_dir, filename):
        conf_mat = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Unseen', 'Seen'], 
                   yticklabels=['Unseen', 'Seen'])
        title = f'Confusion Matrix (threshold={threshold:.2f}, acc={accuracy:.3f})\n'
        title += f'Deep Ensemble: {self.n_models} models'
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()

    def plot_confusion_by_confidence(self, y_test, y_pred_proba, threshold, output_dir, filename):
        y_pred = (y_pred_proba[:, 1] > threshold).astype(int)
        confidence = np.max(y_pred_proba, axis=1)
        
        bins = np.arange(0, 1.1, 0.1)
        bin_labels = [f"{int(b*100)}-{int((b+0.1)*100)}%" for b in bins[:-1]]
        
        n_cols = 5
        n_rows = 2
        _, axes = plt.subplots(n_rows, n_cols, figsize=(20, 8))
        axes = axes.flatten()
        
        for i in range(len(bins) - 1):
            if i == len(bins) - 2:
                mask = (confidence >= bins[i]) & (confidence <= bins[i+1])
            else:
                mask = (confidence >= bins[i]) & (confidence < bins[i+1])
            
            if mask.sum() > 0:
                conf_mat = confusion_matrix(y_test[mask], y_pred[mask], labels=[0, 1])
                conf_mat_pct = conf_mat / conf_mat.sum() * 100
                acc = accuracy_score(y_test[mask], y_pred[mask])
                
                sns.heatmap(conf_mat_pct, annot=True, fmt='.1f', cmap='Blues',
                           xticklabels=['Unseen', 'Seen'], 
                           yticklabels=['Unseen', 'Seen'],
                           ax=axes[i], cbar=False, vmin=0, vmax=100)
                
                axes[i].set_title(f'{bin_labels[i]}\nN={mask.sum()}, Acc={acc:.2%}', 
                                 fontsize=10)
            else:
                axes[i].axis('off')
        
        title = f'Confusion by Confidence (threshold={threshold:.2f})\n'
        title += f'Deep Ensemble: {self.n_models} models'
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_uncertainty_distribution(self, y_test, y_pred_std, optimal_threshold, y_pred_proba, output_dir):
        y_pred = (y_pred_proba[:, 1] > optimal_threshold).astype(int)
        mean_std = y_pred_std.mean(axis=1)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        for cls in [0, 1]:
            mask = y_test == cls
            label = 'Unseen' if cls == 0 else 'Seen'
            axes[0].hist(mean_std[mask], bins=30, alpha=0.6, label=label)
        
        axes[0].set_xlabel('Mean Prediction Uncertainty (std)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Uncertainty Distribution by True Class')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        correct = y_test == y_pred
        axes[1].scatter(mean_std[correct], y_pred_proba[correct, 1], 
                       alpha=0.3, s=10, label='Correct', c='green')
        axes[1].scatter(mean_std[~correct], y_pred_proba[~correct, 1], 
                       alpha=0.3, s=10, label='Wrong', c='red')
        axes[1].axhline(y=optimal_threshold, color='black', linestyle='--', 
                       label=f'Threshold={optimal_threshold:.3f}')
        axes[1].set_xlabel('Prediction Uncertainty (std)')
        axes[1].set_ylabel('Predicted Probability (class 1)')
        axes[1].set_title('Uncertainty vs Prediction Confidence')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'uncertainty_analysis.png'), dpi=300)
        plt.close()
    
    def save_model(self, output_path):
        model_data = {
            'n_models': self.n_models,
            'hidden_dims': self.hidden_dims,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'classes': self.classes_,
            'model_state_dicts': [model.state_dict() for model in self.models]
        }
        torch.save(model_data, output_path)
        print(f"Ensemble model saved to: {output_path}")


# =============================================================================
# Fonctions Optuna
# =============================================================================
def create_model_from_trial(trial: Trial, input_dim: int, num_classes: int = 2):
    """Crée un modèle avec les hyperparamètres suggérés par Optuna."""
    
    # Nombre de couches (1 à 4)
    n_layers = trial.suggest_int('n_layers', 1, 3)
    
    # Dimensions de chaque couche (32 à 1024)
    hidden_dims = []
    for i in range(n_layers):
        dim = trial.suggest_categorical(f'dim_layer_{i}', [32, 64, 128, 256,512])
        hidden_dims.append(dim)
    
    # Dropout
    
    return SingleModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
    ), hidden_dims


def objective(trial: Trial, X_train, y_train, X_test, y_test, device):
    """Optuna objective: AUC on a validation split (``X_test`` / ``y_test`` here are validation data)."""
    
    # Hyperparamètres à optimiser
    n_epochs = trial.suggest_int('n_epochs', 100, 1000, step=50)
    
    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Créer le modèle
    input_dim = X_train_scaled.shape[1]
    model, hidden_dims = create_model_from_trial(trial, input_dim)
    model = model.to(device)
    
    # Stocker les hidden_dims pour récupération ultérieure
    trial.set_user_attr('hidden_dims', hidden_dims)
    
    # Poids de classe
    classes = np.unique(y_train)
    class_weights = compute_sample_weight('balanced', y_train)
    weight_per_class = np.array([class_weights[y_train == c].mean() for c in classes])
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weight_per_class).to(device))
    
    # Tenseurs
    X_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    # Entraînement
    model.train()
    for epoch in range(n_epochs):
        indices = torch.randperm(len(X_tensor))
        
        for start_idx in range(0, len(X_tensor), 128):
            batch_idx = indices[start_idx:start_idx + 128]
            batch_X = X_tensor[batch_idx]
            batch_y = y_tensor[batch_idx]
            
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        
        # Pruning : évaluation intermédiaire pour arrêter les essais peu prometteurs
        if epoch % 50 == 0 and epoch > 0:
            model.eval()
            with torch.no_grad():
                logits = model(X_test_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()
            
            fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
            intermediate_auc = auc(fpr, tpr)
            
            trial.report(intermediate_auc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            model.train()
    
    # Évaluation finale
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    
    fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    return roc_auc


def run_optuna_optimization(X_train, y_train, X_val, y_val, n_trials=50, seed=0):
    """Optuna study; the objective is scored on (X_val, y_val), never on the test set."""
    X_test, y_test = X_val, y_val
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device utilisé: {device}")
    
    # Créer l'étude Optuna (maximiser l'AUC)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100),
        study_name='mlp_memorization_detector'
    )
    
    # Lancer l'optimisation
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_test, y_test, device),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    return study


def train_best_model(study, X_train, y_train, X_test, y_test, output_dir):
    """Entraîne le modèle final avec les meilleurs hyperparamètres."""
    
    best_trial = study.best_trial
    best_params = best_trial.params
    
    # Récupérer hidden_dims depuis user_attrs
    hidden_dims = best_trial.user_attrs['hidden_dims']
    
    print("\n" + "="*60)
    print("MEILLEURS HYPERPARAMÈTRES TROUVÉS:")
    print("="*60)
    print(f"  Nombre de couches: {best_params['n_layers']}")
    print(f"  Dimensions: {hidden_dims}")
    print(f"  Epochs: {best_params['n_epochs']}")
    print(f"\n  MEILLEUR AUC: {study.best_value:.4f}")
    print("="*60 + "\n")
    
    # Créer le classifier avec les meilleurs paramètres
    classifier = DeepEnsembleClassifier(
        n_models=N_MODELS,
        hidden_dims=hidden_dims,
        dropout_rate=0.3
    )
    
    classifier.train(
        X_train, y_train,
        epochs=best_params['n_epochs'],
        lr=5e-4,
    )
    
    # Sauvegarder les résultats Optuna
    optuna_results = {
        'best_params': {
            'n_layers': best_params['n_layers'],
            'hidden_dims': hidden_dims,
            'n_epochs': best_params['n_epochs']
        },
        'best_auc': study.best_value,
        'n_trials': len(study.trials),
        'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'n_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    }
    
    with open(os.path.join(output_dir, 'optuna_results.json'), 'w') as f:
        json.dump(optuna_results, f, indent=2)
    
    return classifier


def plot_optuna_results(study, output_dir):
    """Génère les visualisations Optuna avec matplotlib."""
    
    # 1. Historique d'optimisation
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trial_numbers = [t.number for t in trials]
    values = [t.value for t in trials]
    
    # Meilleur AUC cumulatif
    best_values = []
    current_best = 0
    for v in values:
        current_best = max(current_best, v)
        best_values.append(current_best)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Historique
    axes[0, 0].scatter(trial_numbers, values, alpha=0.6, label='AUC par essai')
    axes[0, 0].plot(trial_numbers, best_values, 'r-', linewidth=2, label='Meilleur AUC')
    axes[0, 0].set_xlabel('Numéro d\'essai')
    axes[0, 0].set_ylabel('AUC')
    axes[0, 0].set_title('Historique d\'optimisation')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Distribution des AUC
    axes[0, 1].hist(values, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(study.best_value, color='red', linestyle='--', 
                       label=f'Meilleur: {study.best_value:.4f}')
    axes[0, 1].set_xlabel('AUC')
    axes[0, 1].set_ylabel('Fréquence')
    axes[0, 1].set_title('Distribution des AUC')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    

    
    # Plot 4: Nombre de couches vs AUC (boxplot)
    n_layers_list = [t.params['n_layers'] for t in trials]
    data_by_layers = {i: [] for i in range(1, 5)}
    for nl, v in zip(n_layers_list, values):
        data_by_layers[nl].append(v)
    
    box_data = [data_by_layers[i] for i in range(1, 5) if data_by_layers[i]]
    box_labels = [str(i) for i in range(1, 5) if data_by_layers[i]]
    
    axes[1, 1].boxplot(box_data, labels=box_labels)
    axes[1, 1].set_xlabel('Nombre de couches')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].set_title('Nombre de couches vs AUC')
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle('Analyse Optuna', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optuna_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot supplémentaire: Importance des paramètres (approximation simple)
    param_names = ['n_layers', 'n_epochs']
    correlations = []
    
    for param in param_names:
        if param in trials[0].params:
            param_values = [t.params[param] for t in trials]
        else:
            param_values = [t.user_attrs.get(param, 0) for t in trials]
        
        if len(set(param_values)) > 1:
            corr = np.corrcoef(param_values, values)[0, 1]
            correlations.append(abs(corr))
        else:
            correlations.append(0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(param_names, correlations, color='steelblue')
    ax.set_xlabel('Corrélation absolue avec AUC')
    ax.set_title('Importance approximative des hyperparamètres')
    ax.grid(alpha=0.3, axis='x')
    
    for bar, corr in zip(bars, correlations):
        ax.text(corr + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{corr:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optuna_param_importance.png'), dpi=300)
    plt.close()
    
    print(f"Visualisations Optuna sauvegardées dans {output_dir}")


# =============================================================================
# Main
# =============================================================================
def split_rows(csv_path, n_train, n_test, seed):
    """Disjoint random train / test rows of a per-sample feature file."""
    df = pd.read_csv(csv_path)
    if len(df) < n_train + n_test:
        raise ValueError(f"{csv_path}: {len(df)} samples, {n_train + n_test} needed")
    idx = np.random.RandomState(seed).permutation(len(df))
    return df.iloc[idx[:n_train]], df.iloc[idx[n_train:n_train + n_test]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recon-dir", required=True, help="output of rf_membership.mia.reconstruction")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-train", type=int, default=1000, help="samples per class for training the MLP")
    parser.add_argument("--n-test", type=int, default=500, help="samples per class for evaluation")
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    sub = os.path.join(args.out_dir, "subsets")
    os.makedirs(sub, exist_ok=True)
    for cls in ("members", "nonmembers"):
        tr, te = split_rows(os.path.join(args.recon_dir, f"{cls}.csv"), args.n_train, args.n_test, args.seed)
        tr.to_csv(os.path.join(sub, f"{cls}_train.csv"), index=False)
        te.to_csv(os.path.join(sub, f"{cls}_test.csv"), index=False)

    csv = lambda cls, split: os.path.join(sub, f"{cls}_{split}.csv")  # noqa: E731
    X_train, y_train, X_test, y_test = DeepEnsembleClassifier().prepare_data_memorization(
        csv("members", "train"), csv("nonmembers", "train"), csv("members", "test"), csv("nonmembers", "test"))
    print(f"Train: {X_train.shape} {dict(Counter(y_train))}  Test: {X_test.shape} {dict(Counter(y_test))}")

    from sklearn.model_selection import train_test_split
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=args.val_fraction, stratify=y_train, random_state=args.seed)

    study = run_optuna_optimization(X_fit, y_fit, X_val, y_val, n_trials=args.n_trials, seed=args.seed)
    plot_optuna_results(study, args.out_dir)
    classifier = train_best_model(study, X_train, y_train, X_test, y_test, args.out_dir)
    results = classifier.evaluate(X_test, y_test, args.out_dir, balance_test=BALANCE_TEST)

    print(f"\nAUC (test): {results['roc_auc']:.4f}")
    print(f"TPR @ FPR: {results['tpr @ fpr']}")
    classifier.save_model(os.path.join(args.out_dir, "mlp_attack.pt"))


if __name__ == "__main__":
    main()
