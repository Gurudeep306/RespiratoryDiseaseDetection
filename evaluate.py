"""
Model evaluation utilities for comprehensive performance analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    classification_report, roc_auc_score
)
from pathlib import Path


class EvaluationVisualizer:
    """Visualizes model performance metrics."""
    
    @staticmethod
    def plot_roc_curve(y_true, y_probs, model_name, save_path=None):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        auc_score = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return auc_score
    
    @staticmethod
    def plot_precision_recall_curve(y_true, y_probs, model_name, save_path=None):
        """Plot precision-recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return pr_auc
    
    @staticmethod
    def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
        """Plot feature importance for tree-based models."""
        if not hasattr(model, 'feature_importances_'):
            print("Model does not support feature importance")
            return
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.title('Top 20 Feature Importances')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_learning_curve(train_scores, val_scores, save_path=None):
        """Plot learning curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_scores, label='Training Score')
        plt.plot(val_scores, label='Validation Score')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


class ComprehensiveEvaluation:
    """Comprehensive model evaluation framework."""
    
    @staticmethod
    def generate_report(y_true, y_pred, y_probs=None, model_name="Model"):
        """Generate comprehensive evaluation report."""
        report = classification_report(y_true, y_pred,
                                      target_names=['HC', 'PD'],
                                      output_dict=True)
        
        report_df = pd.DataFrame(report).transpose()
        
        print(f"\n{'='*60}")
        print(f"{model_name} - Comprehensive Evaluation Report")
        print(f"{'='*60}\n")
        print(report_df)
        
        if y_probs is not None:
            try:
                auc_score = roc_auc_score(y_true, y_probs)
                print(f"\nAUC-ROC Score: {auc_score:.4f}")
            except:
                pass
        
        return report_df
    
    @staticmethod
    def cross_model_comparison(all_metrics):
        """Compare metrics across multiple models."""
        comparison_data = []
        
        for model_name, metrics in all_metrics.items():
            row = {'Model': model_name}
            for metric_name, value in metrics.items():
                if metric_name != 'confusion_matrix':
                    row[metric_name] = value
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by accuracy
        df = df.sort_values('accuracy', ascending=False)
        
        print(f"\n{'='*60}")
        print("Model Comparison")
        print(f"{'='*60}\n")
        print(df.to_string(index=False))
        
        return df
    
    @staticmethod
    def save_evaluation_report(all_metrics, output_dir, include_visualizations=True):
        """Save complete evaluation report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create summary DataFrame
        summary_data = []
        for model_name, metrics in all_metrics.items():
            row = {'Model': model_name}
            for metric_name, value in metrics.items():
                if metric_name != 'confusion_matrix':
                    row[metric_name] = f"{value:.2f}"
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = output_dir / 'evaluation_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        
        print(f"Evaluation summary saved to {summary_path}")
        
        if include_visualizations:
            # Create comparison plots
            EvaluationVisualizer.plot_model_comparison(all_metrics, output_dir / 'comparison.png')
            EvaluationVisualizer.plot_confusion_matrices(all_metrics, output_dir)
        
        return summary_path
    
    @staticmethod
    def plot_model_comparison(all_metrics, save_path):
        """Plot model performance comparison."""
        metrics_list = []
        for model_name, metrics in all_metrics.items():
            for metric_name, value in metrics.items():
                if metric_name != 'confusion_matrix':
                    metrics_list.append({
                        'Model': model_name,
                        'Metric': metric_name,
                        'Value': value
                    })
        
        df = pd.DataFrame(metrics_list)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(data=df, x='Metric', y='Value', hue='Model', ax=ax)
        ax.set_ylim(0, 105)
        ax.set_ylabel('Score (%)')
        ax.set_title('Model Performance Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_confusion_matrices(all_metrics, output_dir):
        """Plot confusion matrices for all models."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, metrics in all_metrics.items():
            if 'confusion_matrix' in metrics:
                cm = metrics['confusion_matrix']
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=['HC', 'PD'],
                           yticklabels=['HC', 'PD'],
                           cbar_kws={'label': 'Count'})
                plt.title(f'Confusion Matrix - {model_name}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                
                save_path = output_dir / f'{model_name.lower().replace(" ", "_")}_cm.png'
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()


class PerformanceAnalyzer:
    """Analyzes model performance patterns."""
    
    @staticmethod
    def analyze_prediction_confidence(y_true, y_probs):
        """Analyze prediction confidence patterns."""
        confidence = np.abs(y_probs - 0.5) * 2  # 0 to 1 scale
        
        analysis = {
            'mean_confidence': np.mean(confidence),
            'std_confidence': np.std(confidence),
            'min_confidence': np.min(confidence),
            'max_confidence': np.max(confidence),
            'high_confidence_rate': np.mean(confidence > 0.8),
            'uncertain_rate': np.mean((confidence > 0.3) & (confidence < 0.7))
        }
        
        return analysis
    
    @staticmethod
    def analyze_class_imbalance(y_true):
        """Analyze class imbalance in dataset."""
        unique, counts = np.unique(y_true, return_counts=True)
        
        analysis = {}
        for label, count in zip(unique, counts):
            class_name = 'PD' if label == 1 else 'HC'
            analysis[class_name] = {
                'count': count,
                'percentage': count / len(y_true) * 100
            }
        
        return analysis
    
    @staticmethod
    def get_misclassified_samples(y_true, y_pred):
        """Get indices of misclassified samples."""
        misclassified_indices = np.where(y_true != y_pred)[0]
        return misclassified_indices
    
    @staticmethod
    def analyze_threshold_effect(y_true, y_probs, thresholds=None):
        """Analyze effect of classification threshold."""
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
        
        results = []
        for threshold in thresholds:
            y_pred_threshold = (y_probs > threshold).astype(int)
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            try:
                accuracy = accuracy_score(y_true, y_pred_threshold)
                precision = precision_score(y_true, y_pred_threshold, zero_division=0)
                recall = recall_score(y_true, y_pred_threshold, zero_division=0)
                
                results.append({
                    'threshold': threshold,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': 2 * (precision * recall) / (precision + recall + 1e-8)
                })
            except:
                pass
        
        return pd.DataFrame(results)
