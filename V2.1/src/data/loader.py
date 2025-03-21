"""
Data loading and preprocessing module.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from ..config.settings import RAW_DATA_DIR, CSV_ENCODING

class DataLoader:
    def __init__(self):
        """Initialize the data loader."""
        self.raw_data_dir = RAW_DATA_DIR

    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame.
        
        Args:
            filename: Name of the CSV file
            
        Returns:
            DataFrame containing the CSV data
        """
        file_path = self.raw_data_dir / filename
        return pd.read_csv(file_path, encoding=CSV_ENCODING)

    def load_multiple_csvs(self, filenames: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load multiple CSV files into a dictionary of DataFrames.
        
        Args:
            filenames: List of CSV filenames
            
        Returns:
            Dictionary mapping filenames to DataFrames
        """
        return {
            filename: self.load_csv(filename)
            for filename in filenames
        }

    def preprocess_feedback_data(
        self,
        df: pd.DataFrame,
        group_col: str = "group",
        student_col: str = "student",
        feedback_col: str = "feedback",
        date_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Preprocess feedback data into a structured format.
        
        Args:
            df: Input DataFrame
            group_col: Name of the group column
            student_col: Name of the student column
            feedback_col: Name of the feedback column
            date_col: Optional name of the date column
            
        Returns:
            Structured dictionary of feedback data
        """
        # Group by group and student
        grouped_data = df.groupby([group_col, student_col])[feedback_col].agg(list).reset_index()
        
        # Create nested structure
        processed_data = {}
        for _, row in grouped_data.iterrows():
            group = row[group_col]
            student = row[student_col]
            feedback = row[feedback_col]
            
            if group not in processed_data:
                processed_data[group] = {}
            
            processed_data[group][student] = {
                "feedback": feedback,
                "feedback_count": len(feedback)
            }
            
            if date_col:
                dates = df[
                    (df[group_col] == group) & 
                    (df[student_col] == student)
                ][date_col].tolist()
                processed_data[group][student]["dates"] = dates
        
        return processed_data

    def get_class_summary(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of class-level statistics.
        
        Args:
            processed_data: Processed feedback data
            
        Returns:
            Dictionary containing class-level statistics
        """
        total_groups = len(processed_data)
        total_students = sum(len(group_data) for group_data in processed_data.values())
        total_feedback = sum(
            student_data["feedback_count"]
            for group_data in processed_data.values()
            for student_data in group_data.values()
        )
        
        return {
            "total_groups": total_groups,
            "total_students": total_students,
            "total_feedback": total_feedback,
            "average_feedback_per_student": total_feedback / total_students if total_students > 0 else 0
        }

    def get_group_summary(
        self,
        processed_data: Dict[str, Any],
        group_name: str
    ) -> Dict[str, Any]:
        """
        Generate a summary for a specific group.
        
        Args:
            processed_data: Processed feedback data
            group_name: Name of the group to analyze
            
        Returns:
            Dictionary containing group-level statistics
        """
        if group_name not in processed_data:
            raise ValueError(f"Group {group_name} not found in data")
            
        group_data = processed_data[group_name]
        total_students = len(group_data)
        total_feedback = sum(
            student_data["feedback_count"]
            for student_data in group_data.values()
        )
        
        return {
            "total_students": total_students,
            "total_feedback": total_feedback,
            "average_feedback_per_student": total_feedback / total_students if total_students > 0 else 0
        }

    def get_student_data(
        self,
        processed_data: Dict[str, Any],
        group_name: str,
        student_name: str
    ) -> Dict[str, Any]:
        """
        Get data for a specific student.
        
        Args:
            processed_data: Processed feedback data
            group_name: Name of the group
            student_name: Name of the student
            
        Returns:
            Dictionary containing student-level data
        """
        if group_name not in processed_data:
            raise ValueError(f"Group {group_name} not found in data")
            
        if student_name not in processed_data[group_name]:
            raise ValueError(f"Student {student_name} not found in group {group_name}")
            
        return processed_data[group_name][student_name] 