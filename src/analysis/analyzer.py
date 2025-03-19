"""
Main analysis module for coordinating feedback analysis.
"""
from typing import Dict, List, Any, Optional
from ..data.loader import DataLoader
from ..models.llm import LLMInterface
from ..models.evaluator import FeedbackEvaluator
from ..config.settings import DEFAULT_FRAMEWORK, DEFAULT_MODEL

class FeedbackAnalyzer:
    def __init__(self, model_key: str = DEFAULT_MODEL):
        """
        Initialize the feedback analyzer.
        
        Args:
            model_key: Key of the model configuration to use
        """
        self.data_loader = DataLoader()
        self.llm = LLMInterface(model_key=model_key)
        self.evaluator = FeedbackEvaluator()

    def analyze_class_feedback(
        self,
        csv_files: List[str],
        group_col: str = "group",
        student_col: str = "student",
        feedback_col: str = "feedback",
        date_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze feedback at the class level.
        
        Args:
            csv_files: List of CSV files to analyze
            group_col: Name of the group column
            student_col: Name of the student column
            feedback_col: Name of the feedback column
            date_col: Optional name of the date column
            
        Returns:
            Dictionary containing class-level analysis
        """
        # Load and preprocess data
        dfs = self.data_loader.load_multiple_csvs(csv_files)
        processed_data = {}
        
        for df in dfs.values():
            processed = self.data_loader.preprocess_feedback_data(
                df,
                group_col,
                student_col,
                feedback_col,
                date_col
            )
            processed_data.update(processed)
        
        # Get basic statistics
        stats = self.data_loader.get_class_summary(processed_data)
        
        # Get LLM analysis
        llm_analysis = self.llm.analyze_class_level(processed_data)
        
        return {
            "statistics": stats,
            "llm_analysis": llm_analysis,
            "model_used": self.llm.model_name
        }

    def analyze_group_feedback(
        self,
        csv_files: List[str],
        group_name: str,
        group_col: str = "group",
        student_col: str = "student",
        feedback_col: str = "feedback",
        date_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze feedback at the group level.
        
        Args:
            csv_files: List of CSV files to analyze
            group_name: Name of the group to analyze
            group_col: Name of the group column
            student_col: Name of the student column
            feedback_col: Name of the feedback column
            date_col: Optional name of the date column
            
        Returns:
            Dictionary containing group-level analysis
        """
        # Load and preprocess data
        dfs = self.data_loader.load_multiple_csvs(csv_files)
        processed_data = {}
        
        for df in dfs.values():
            processed = self.data_loader.preprocess_feedback_data(
                df,
                group_col,
                student_col,
                feedback_col,
                date_col
            )
            processed_data.update(processed)
        
        # Get basic statistics
        stats = self.data_loader.get_group_summary(processed_data, group_name)
        
        # Get group data
        group_data = processed_data[group_name]
        
        # Get LLM analysis
        llm_analysis = self.llm.analyze_group_level(group_data)
        
        return {
            "statistics": stats,
            "llm_analysis": llm_analysis,
            "model_used": self.llm.model_name
        }

    def analyze_student_feedback(
        self,
        csv_files: List[str],
        group_name: str,
        student_name: str,
        framework: str = DEFAULT_FRAMEWORK,
        group_col: str = "group",
        student_col: str = "student",
        feedback_col: str = "feedback",
        date_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze feedback at the student level.
        
        Args:
            csv_files: List of CSV files to analyze
            group_name: Name of the group
            student_name: Name of the student
            framework: Analysis framework to use
            group_col: Name of the group column
            student_col: Name of the student column
            feedback_col: Name of the feedback column
            date_col: Optional name of the date column
            
        Returns:
            Dictionary containing student-level analysis
        """
        # Load and preprocess data
        dfs = self.data_loader.load_multiple_csvs(csv_files)
        processed_data = {}
        
        for df in dfs.values():
            processed = self.data_loader.preprocess_feedback_data(
                df,
                group_col,
                student_col,
                feedback_col,
                date_col
            )
            processed_data.update(processed)
        
        # Get student data
        student_data = self.data_loader.get_student_data(
            processed_data,
            group_name,
            student_name
        )
        
        # Get LLM analysis
        llm_analysis = self.llm.analyze_student_level(
            student_data,
            framework
        )
        
        # Evaluate feedback quality
        evaluation_results = []
        for feedback in student_data["feedback"]:
            evaluation = self.evaluator.evaluate_feedback(feedback)
            evaluation_results.append(evaluation)
        
        return {
            "student_data": student_data,
            "llm_analysis": llm_analysis,
            "evaluation_results": evaluation_results,
            "model_used": self.llm.model_name
        }

    def compare_feedback_quality(
        self,
        csv_files: List[str],
        group_name: str,
        student_name: str,
        group_col: str = "group",
        student_col: str = "student",
        feedback_col: str = "feedback"
    ) -> Dict[str, Any]:
        """
        Compare feedback quality across different sources.
        
        Args:
            csv_files: List of CSV files to analyze
            group_name: Name of the group
            student_name: Name of the student
            group_col: Name of the group column
            student_col: Name of the student column
            feedback_col: Name of the feedback column
            
        Returns:
            Dictionary containing comparison results
        """
        # Get student feedback
        student_analysis = self.analyze_student_feedback(
            csv_files,
            group_name,
            student_name,
            group_col=group_col,
            student_col=student_col,
            feedback_col=feedback_col
        )
        
        # Compare evaluations
        comparison = self.evaluator.compare_evaluations(
            student_analysis["evaluation_results"]
        )
        
        return {
            "student_analysis": student_analysis,
            "comparison": comparison,
            "model_used": self.llm.model_name
        } 