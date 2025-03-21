"""
Tests for the data loader component.
"""
import pytest
import pandas as pd
from src.data.loader import DataLoader

def test_load_csv(sample_csv_file, data_loader):
    """Test loading a CSV file."""
    df = data_loader.load_csv(sample_csv_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert all(col in df.columns for col in ['group', 'student', 'feedback', 'date'])

def test_preprocess_feedback_data(sample_feedback_data, data_loader):
    """Test preprocessing feedback data."""
    processed = data_loader.preprocess_feedback_data(
        sample_feedback_data,
        group_col='group',
        student_col='student',
        feedback_col='feedback',
        date_col='date'
    )
    
    assert isinstance(processed, dict)
    assert 'Group A' in processed
    assert 'Group B' in processed
    assert len(processed['Group A']['students']) == 2
    assert len(processed['Group B']['students']) == 2

def test_get_class_summary(sample_feedback_data, data_loader):
    """Test getting class-level summary."""
    processed = data_loader.preprocess_feedback_data(
        sample_feedback_data,
        group_col='group',
        student_col='student',
        feedback_col='feedback',
        date_col='date'
    )
    
    summary = data_loader.get_class_summary(processed)
    assert isinstance(summary, dict)
    assert 'total_students' in summary
    assert 'total_groups' in summary
    assert 'total_feedback' in summary
    assert summary['total_students'] == 4
    assert summary['total_groups'] == 2

def test_get_group_summary(sample_feedback_data, data_loader):
    """Test getting group-level summary."""
    processed = data_loader.preprocess_feedback_data(
        sample_feedback_data,
        group_col='group',
        student_col='student',
        feedback_col='feedback',
        date_col='date'
    )
    
    summary = data_loader.get_group_summary(processed, 'Group A')
    assert isinstance(summary, dict)
    assert 'group_name' in summary
    assert 'student_count' in summary
    assert 'feedback_count' in summary
    assert summary['student_count'] == 2
    assert summary['feedback_count'] == 2

def test_get_student_data(sample_feedback_data, data_loader):
    """Test getting student-level data."""
    processed = data_loader.preprocess_feedback_data(
        sample_feedback_data,
        group_col='group',
        student_col='student',
        feedback_col='feedback',
        date_col='date'
    )
    
    student_data = data_loader.get_student_data(
        processed,
        'Group A',
        'Student 1'
    )
    
    assert isinstance(student_data, dict)
    assert 'student_name' in student_data
    assert 'group_name' in student_data
    assert 'feedback' in student_data
    assert len(student_data['feedback']) == 1 