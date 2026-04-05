"""Evaluation metrics for speaker verification."""

from typing import Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d


class EqualErrorRate:
    """Equal Error Rate (EER) and related metrics."""
    
    @staticmethod
    def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
        """
        Compute Equal Error Rate (EER) and threshold.
        
        Args:
            y_true: Binary labels (1: same speaker, 0: different)
            y_scores: Similarity scores
            
        Returns:
            Tuple of (EER, threshold)
        """
        # Sort by scores
        sorted_indices = np.argsort(-y_scores)
        y_true_sorted = y_true[sorted_indices]
        
        # Compute FPR and FNR
        n_positive = np.sum(y_true)
        n_negative = len(y_true) - n_positive
        
        fp = np.cumsum(np.logical_not(y_true_sorted))
        fn = np.cumsum(np.logical_not(y_true_sorted))
        
        # Alternative calculation
        tp = np.cumsum(y_true_sorted)
        tn = n_negative - fp
        
        fpr = fp / n_negative if n_negative > 0 else np.zeros_like(fp)
        fnr = (n_positive - tp) / n_positive if n_positive > 0 else np.zeros_like(tp)
        
        # Find EER
        eer_threshold = None
        min_diff = float('inf')
        
        for i, (f_p, f_n) in enumerate(zip(fpr, fnr)):
            diff = abs(f_p - f_n)
            if diff < min_diff:
                min_diff = diff
                eer_threshold = y_scores[sorted_indices[i]]
        
        eer = min_diff
        
        return eer, eer_threshold
    
    @staticmethod
    def compute_far_frr(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        threshold: float
    ) -> Tuple[float, float]:
        """
        Compute False Acceptance Rate (FAR) and False Rejection Rate (FRR).
        
        Args:
            y_true: Binary labels (1: same speaker, 0: different)
            y_scores: Similarity scores
            threshold: Decision threshold
            
        Returns:
            Tuple of (FAR, FRR)
        """
        predictions = (y_scores > threshold).astype(int)
        
        # FAR: False positives / negatives
        false_positives = np.sum((predictions == 1) & (y_true == 0))
        n_negative = np.sum(y_true == 0)
        far = false_positives / n_negative if n_negative > 0 else 0.0
        
        # FRR: False negatives / positives
        false_negatives = np.sum((predictions == 0) & (y_true == 1))
        n_positive = np.sum(y_true == 1)
        frr = false_negatives / n_positive if n_positive > 0 else 0.0
        
        return far, frr


class DetectionErrorTrade:
    """Detection Error Trade-off (DET) curve."""
    
    @staticmethod
    def compute_det(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        num_thresholds: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute DET curve points.
        
        Args:
            y_true: Binary labels
            y_scores: Similarity scores
            num_thresholds: Number of thresholds to evaluate
            
        Returns:
            Tuple of (FPR, FNR, thresholds)
        """
        thresholds = np.linspace(np.min(y_scores), np.max(y_scores), num_thresholds)
        
        far_list = []
        frr_list = []
        
        for threshold in thresholds:
            far, frr = EqualErrorRate.compute_far_frr(y_true, y_scores, threshold)
            far_list.append(far)
            frr_list.append(frr)
        
        return np.array(far_list), np.array(frr_list), thresholds


class AccuracyMetrics:
    """Accuracy and related metrics."""
    
    @staticmethod
    def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute accuracy.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Accuracy
        """
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def compute_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute precision (only for binary case).
        
        Args:
            y_true: True labels (binary)
            y_pred: Predicted labels (binary)
            
        Returns:
            Precision
        """
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        predicted_positives = np.sum(y_pred == 1)
        
        return true_positives / predicted_positives if predicted_positives > 0 else 0.0
    
    @staticmethod
    def compute_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute recall (sensitivity/TPR).
        
        Args:
            y_true: True labels (binary)
            y_pred: Predicted labels (binary)
            
        Returns:
            Recall
        """
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        positives = np.sum(y_true == 1)
        
        return true_positives / positives if positives > 0 else 0.0
    
    @staticmethod
    def compute_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute F1 score.
        
        Args:
            y_true: True labels (binary)
            y_pred: Predicted labels (binary)
            
        Returns:
            F1 score
        """
        precision = AccuracyMetrics.compute_precision(y_true, y_pred)
        recall = AccuracyMetrics.compute_recall(y_true, y_pred)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """
        Compute ROC AUC score.
        
        Args:
            y_true: True labels (binary)
            y_scores: Similarity scores
            
        Returns:
            ROC AUC
        """
        # Sort by scores
        sorted_indices = np.argsort(-y_scores)
        y_true_sorted = y_true[sorted_indices]
        
        # Compute cumulative sums
        n_positive = np.sum(y_true)
        n_negative = len(y_true) - n_positive
        
        tp = np.cumsum(y_true_sorted)
        fp = np.cumsum(1 - y_true_sorted)
        
        tpr = tp / n_positive if n_positive > 0 else np.zeros_like(tp)
        fpr = fp / n_negative if n_negative > 0 else np.zeros_like(fp)
        
        # Add corner points
        tpr = np.concatenate(([0], tpr, [1]))
        fpr = np.concatenate(([0], fpr, [1]))
        
        # Compute AUC using trapezoidal rule
        auc = np.trapz(tpr, fpr)
        
        return auc


class SpeakerVerificationMetrics:
    """Combined metrics for speaker verification."""
    
    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        threshold: float = 0.5
    ) -> dict:
        """
        Compute all evaluation metrics.
        
        Args:
            y_true: Binary labels (1: same speaker, 0: different)
            y_scores: Similarity scores
            threshold: Decision threshold
            
        Returns:
            Dictionary with all metrics
        """
        # Binary predictions
        y_pred = (y_scores > threshold).astype(int)
        
        # EER
        eer, eer_threshold = EqualErrorRate.compute_eer(y_true, y_scores)
        
        # FAR/FRR
        far, frr = EqualErrorRate.compute_far_frr(y_true, y_scores, threshold)
        
        # Accuracy metrics
        accuracy = AccuracyMetrics.compute_accuracy(y_true, y_pred)
        precision = AccuracyMetrics.compute_precision(y_true, y_pred)
        recall = AccuracyMetrics.compute_recall(y_true, y_pred)
        f1 = AccuracyMetrics.compute_f1_score(y_true, y_pred)
        auc = AccuracyMetrics.compute_roc_auc(y_true, y_scores)
        
        return {
            'eer': eer,
            'eer_threshold': eer_threshold,
            'far': far,
            'frr': frr,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
