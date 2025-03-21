"""
Test configuration and fixtures.
"""
import pytest
import pandas as pd
from pathlib import Path
import json
from src.config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.models.llm import LLMInterface
from src.data.loader import DataLoader
from src.models.evaluator import FeedbackEvaluator
from src.analysis.analyzer import FeedbackAnalyzer

@pytest.fixture
def sample_feedback_data():
    """Create sample feedback data for testing."""
    data = {
        'group': ['Group A', 'Group A', 'Group B', 'Group B'],
        'student': ['Student 1', 'Student 2', 'Student 3', 'Student 4'],
        'feedback': [
            'Great work on the project!',
            'Could improve documentation.',
            'Excellent presentation skills.',
            'Needs more attention to detail.'
        ],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_csv_file(sample_feedback_data):
    """Create a sample CSV file for testing."""
    file_path = RAW_DATA_DIR / "test_feedback.csv"
    sample_feedback_data.to_csv(file_path, index=False)
    yield file_path
    file_path.unlink(missing_ok=True)

@pytest.fixture
def llm_interface():
    """Create an LLM interface instance for testing."""
    return LLMInterface(model_key="llama-3.1-8b")

@pytest.fixture
def data_loader():
    """Create a data loader instance for testing."""
    return DataLoader()

@pytest.fixture
def feedback_evaluator():
    """Create a feedback evaluator instance for testing."""
    return FeedbackEvaluator()

@pytest.fixture
def feedback_analyzer():
    """Create a feedback analyzer instance for testing."""
    return FeedbackAnalyzer(model_key="llama-3.1-8b") 