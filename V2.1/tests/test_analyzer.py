"""
Tests for the feedback analyzer component.
"""
import pytest
from src.analysis.analyzer import FeedbackAnalyzer

def test_analyzer_initialization(feedback_analyzer):
    """Test feedback analyzer initialization."""
    assert feedback_analyzer.llm is not None
    assert feedback_analyzer.data_loader is not None
    assert feedback_analyzer.evaluator is not None

def test_analyze_class_feedback(feedback_analyzer, sample_csv_file):
    """Test class-level feedback analysis."""
    analysis = feedback_analyzer.analyze_class_feedback(
        [str(sample_csv_file)],
        group_col='group',
        student_col='student',
        feedback_col='feedback',
        date_col='date'
    )
    
    assert isinstance(analysis, dict)
    assert 'statistics' in analysis
    assert 'llm_analysis' in analysis
    assert 'model_used' in analysis
    assert analysis['model_used'] == "meta-llama/Llama-3.1-8B-Instruct"

def test_analyze_group_feedback(feedback_analyzer, sample_csv_file):
    """Test group-level feedback analysis."""
    analysis = feedback_analyzer.analyze_group_feedback(
        [str(sample_csv_file)],
        group_name='Group A',
        group_col='group',
        student_col='student',
        feedback_col='feedback',
        date_col='date'
    )
    
    assert isinstance(analysis, dict)
    assert 'statistics' in analysis
    assert 'llm_analysis' in analysis
    assert 'model_used' in analysis
    assert analysis['statistics']['group_name'] == 'Group A'

def test_analyze_student_feedback(feedback_analyzer, sample_csv_file):
    """Test student-level feedback analysis."""
    analysis = feedback_analyzer.analyze_student_feedback(
        [str(sample_csv_file)],
        group_name='Group A',
        student_name='Student 1',
        framework='ICAP',
        group_col='group',
        student_col='student',
        feedback_col='feedback',
        date_col='date'
    )
    
    assert isinstance(analysis, dict)
    assert 'student_data' in analysis
    assert 'llm_analysis' in analysis
    assert 'evaluation_results' in analysis
    assert 'model_used' in analysis
    assert analysis['student_data']['student_name'] == 'Student 1'

def test_compare_feedback_quality(feedback_analyzer, sample_csv_file):
    """Test feedback quality comparison."""
    comparison = feedback_analyzer.compare_feedback_quality(
        [str(sample_csv_file)],
        group_name='Group A',
        student_name='Student 1',
        group_col='group',
        student_col='student',
        feedback_col='feedback'
    )
    
    assert isinstance(comparison, dict)
    assert 'student_analysis' in comparison
    assert 'comparison' in comparison
    assert 'model_used' in comparison
    assert isinstance(comparison['comparison'], dict) 