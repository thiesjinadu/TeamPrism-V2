"""
Tests for the API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app
import pandas as pd
from pathlib import Path
from src.config.settings import RAW_DATA_DIR

client = TestClient(app)

def test_health_check():
    """Test health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_get_available_models():
    """Test getting available models endpoint."""
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, dict)
    assert 'llama-3.1-8b' in models
    assert 'llama-3.2-1b' in models

def test_upload_feedback(sample_csv_file):
    """Test feedback file upload endpoint."""
    with open(sample_csv_file, "rb") as f:
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test_feedback.csv", f, "text/csv")}
        )
    assert response.status_code == 200
    assert "message" in response.json()

def test_class_analysis(sample_csv_file):
    """Test class-level analysis endpoint."""
    response = client.get(
        "/api/v1/class-analysis",
        params={
            "files": [str(sample_csv_file)],
            "model_key": "llama-3.1-8b",
            "group_col": "group",
            "student_col": "student",
            "feedback_col": "feedback",
            "date_col": "date"
        }
    )
    assert response.status_code == 200
    analysis = response.json()
    assert "statistics" in analysis
    assert "llm_analysis" in analysis
    assert "model_used" in analysis

def test_group_analysis(sample_csv_file):
    """Test group-level analysis endpoint."""
    response = client.get(
        "/api/v1/group-analysis",
        params={
            "files": [str(sample_csv_file)],
            "group_name": "Group A",
            "model_key": "llama-3.1-8b",
            "group_col": "group",
            "student_col": "student",
            "feedback_col": "feedback",
            "date_col": "date"
        }
    )
    assert response.status_code == 200
    analysis = response.json()
    assert "statistics" in analysis
    assert "llm_analysis" in analysis
    assert "model_used" in analysis
    assert analysis["statistics"]["group_name"] == "Group A"

def test_student_analysis(sample_csv_file):
    """Test student-level analysis endpoint."""
    response = client.get(
        "/api/v1/student-analysis",
        params={
            "files": [str(sample_csv_file)],
            "group_name": "Group A",
            "student_name": "Student 1",
            "model_key": "llama-3.1-8b",
            "framework": "ICAP",
            "group_col": "group",
            "student_col": "student",
            "feedback_col": "feedback",
            "date_col": "date"
        }
    )
    assert response.status_code == 200
    analysis = response.json()
    assert "student_data" in analysis
    assert "llm_analysis" in analysis
    assert "evaluation_results" in analysis
    assert "model_used" in analysis
    assert analysis["student_data"]["student_name"] == "Student 1"

def test_compare_feedback(sample_csv_file):
    """Test feedback comparison endpoint."""
    response = client.get(
        "/api/v1/compare-feedback",
        params={
            "files": [str(sample_csv_file)],
            "group_name": "Group A",
            "student_name": "Student 1",
            "model_key": "llama-3.1-8b",
            "group_col": "group",
            "student_col": "student",
            "feedback_col": "feedback"
        }
    )
    assert response.status_code == 200
    comparison = response.json()
    assert "student_analysis" in comparison
    assert "comparison" in comparison
    assert "model_used" in comparison 