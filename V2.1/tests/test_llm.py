"""
Tests for the LLM interface component.
"""
import pytest
from src.models.llm import LLMInterface

def test_llm_initialization(llm_interface):
    """Test LLM interface initialization."""
    assert llm_interface.model_name == "meta-llama/Llama-3.1-8B-Instruct"
    assert llm_interface.max_tokens == 512
    assert llm_interface.temperature == 0.7
    assert llm_interface.context_length == 4096

def test_generate_response(llm_interface):
    """Test generating a response from the LLM."""
    prompt = "What is the capital of France?"
    response = llm_interface.generate_response(prompt)
    assert isinstance(response, str)
    assert len(response) > 0

def test_analyze_class_level(llm_interface, sample_feedback_data):
    """Test class-level analysis."""
    processed_data = {
        'Group A': {
            'students': ['Student 1', 'Student 2'],
            'feedback': ['Great work!', 'Good job!']
        },
        'Group B': {
            'students': ['Student 3', 'Student 4'],
            'feedback': ['Excellent!', 'Well done!']
        }
    }
    
    analysis = llm_interface.analyze_class_level(processed_data)
    assert isinstance(analysis, dict)
    assert 'summary' in analysis
    assert 'key_insights' in analysis
    assert 'recommendations' in analysis

def test_analyze_group_level(llm_interface):
    """Test group-level analysis."""
    group_data = {
        'students': ['Student 1', 'Student 2'],
        'feedback': ['Great work!', 'Good job!']
    }
    
    analysis = llm_interface.analyze_group_level(group_data)
    assert isinstance(analysis, dict)
    assert 'group_summary' in analysis
    assert 'student_performance' in analysis
    assert 'improvement_areas' in analysis

def test_analyze_student_level(llm_interface):
    """Test student-level analysis."""
    student_data = {
        'student_name': 'Student 1',
        'group_name': 'Group A',
        'feedback': ['Great work!', 'Good job!']
    }
    
    analysis = llm_interface.analyze_student_level(student_data, framework='ICAP')
    assert isinstance(analysis, dict)
    assert 'student_summary' in analysis
    assert 'learning_style' in analysis
    assert 'improvement_suggestions' in analysis

def test_evaluate_feedback(llm_interface):
    """Test feedback evaluation."""
    feedback = "Great work on the project! Your presentation was clear and well-structured."
    evaluation = llm_interface.evaluate_feedback(feedback)
    assert isinstance(evaluation, dict)
    assert 'quality_score' in evaluation
    assert 'strengths' in evaluation
    assert 'areas_for_improvement' in evaluation

def test_get_available_models():
    """Test getting available model configurations."""
    models = LLMInterface.get_available_models()
    assert isinstance(models, dict)
    assert 'llama-3.1-8b' in models
    assert 'llama-3.2-1b' in models 