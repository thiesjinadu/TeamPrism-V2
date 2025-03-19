"""
Evaluation module for feedback analysis.
"""
from typing import Dict, List, Any, Optional
import numpy as np
from bert_score import score
from transformers import AutoTokenizer, AutoModel
import torch
from ..config.settings import EVALUATION_METRICS
from .llm import LLMInterface

class FeedbackEvaluator:
    def __init__(self):
        """Initialize the feedback evaluator."""
        self.llm = LLMInterface()
        self.metrics = EVALUATION_METRICS

    def evaluate_with_bertscore(
        self,
        references: List[str],
        candidates: List[str],
        model_type: str = "microsoft/deberta-xlarge-mnli"
    ) -> Dict[str, float]:
        """
        Evaluate feedback using BERTScore.
        
        Args:
            references: List of reference feedback texts
            candidates: List of candidate feedback texts
            model_type: Type of BERT model to use
            
        Returns:
            Dictionary containing BERTScore metrics
        """
        P, R, F1 = score(
            candidates,
            references,
            lang="en",
            model_type=model_type,
            verbose=True
        )
        
        return {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item()
        }

    def evaluate_with_lime(
        self,
        text: str,
        model: Any,
        num_features: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Evaluate feedback using LIME.
        
        Args:
            text: Input text to evaluate
            model: Model to explain
            num_features: Number of features to consider
            
        Returns:
            Dictionary containing LIME explanations
        """
        from lime import lime_text
        from lime.lime_text import LimeTextExplainer
        
        explainer = LimeTextExplainer(class_names=['Positive', 'Negative'])
        exp = explainer.explain_instance(text, model.predict_proba)
        
        return {
            "explanations": exp.as_list(),
            "top_features": exp.as_list()[:num_features]
        }

    def evaluate_with_shap(
        self,
        text: str,
        model: Any,
        background_data: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate feedback using SHAP.
        
        Args:
            text: Input text to evaluate
            model: Model to explain
            background_data: Background dataset for SHAP
            
        Returns:
            Dictionary containing SHAP values
        """
        import shap
        
        # Create a background dataset
        background = shap.sample(background_data, 100)
        
        # Create an explainer
        explainer = shap.KernelExplainer(model.predict_proba, background)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(text)
        
        return {
            "shap_values": shap_values.tolist(),
            "feature_importance": np.abs(shap_values).mean(axis=0).tolist()
        }

    def evaluate_with_llm(
        self,
        feedback_text: str
    ) -> Dict[str, Any]:
        """
        Evaluate feedback using LLM.
        
        Args:
            feedback_text: Text to evaluate
            
        Returns:
            Dictionary containing LLM evaluation results
        """
        return self.llm.evaluate_feedback(feedback_text)

    def evaluate_feedback(
        self,
        feedback_text: str,
        reference_text: Optional[str] = None,
        model: Optional[Any] = None,
        background_data: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate feedback using all configured metrics.
        
        Args:
            feedback_text: Text to evaluate
            reference_text: Optional reference text for BERTScore
            model: Optional model for LIME/SHAP
            background_data: Optional background data for SHAP
            
        Returns:
            Dictionary containing all evaluation results
        """
        results = {}
        
        if "bertscore" in self.metrics and reference_text:
            results["bertscore"] = self.evaluate_with_bertscore(
                [reference_text],
                [feedback_text]
            )
            
        if "lime" in self.metrics and model:
            results["lime"] = self.evaluate_with_lime(feedback_text, model)
            
        if "shap" in self.metrics and model and background_data:
            results["shap"] = self.evaluate_with_shap(
                feedback_text,
                model,
                background_data
            )
            
        if "llm_evaluation" in self.metrics:
            results["llm_evaluation"] = self.evaluate_with_llm(feedback_text)
            
        return results

    def compare_evaluations(
        self,
        evaluations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare multiple evaluations.
        
        Args:
            evaluations: List of evaluation results
            
        Returns:
            Dictionary containing comparison metrics
        """
        comparison = {}
        
        # Compare BERTScore results if available
        if all("bertscore" in eval for eval in evaluations):
            comparison["bertscore"] = {
                "precision": [eval["bertscore"]["precision"] for eval in evaluations],
                "recall": [eval["bertscore"]["recall"] for eval in evaluations],
                "f1": [eval["bertscore"]["f1"] for eval in evaluations]
            }
            
        # Compare LLM evaluation scores if available
        if all("llm_evaluation" in eval for eval in evaluations):
            comparison["llm_scores"] = [
                eval["llm_evaluation"]["score"]
                for eval in evaluations
            ]
            
        return comparison 