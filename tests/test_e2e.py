"""
End-to-end tests for the feedback analysis system.
"""
import pytest
import pandas as pd
from pathlib import Path
from src.config.settings import RAW_DATA_DIR
from src.analysis.analyzer import FeedbackAnalyzer
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

@pytest.fixture
def sample_feedback_data():
    """Create sample feedback data for end-to-end testing."""
    data = {
        'group': ['Group A', 'Group A', 'Group B', 'Group B', 'Group A'],
        'student': ['Student 1', 'Student 2', 'Student 3', 'Student 4', 'Student 1'],
        'feedback': [
            'Great work on the project! Your presentation was clear and well-structured.',
            'Could improve documentation. Consider adding more comments to the code.',
            'Excellent presentation skills. The demo was very engaging.',
            'Needs more attention to detail in the implementation.',
            'Good progress on the latest features. Keep up the good work!'
        ],
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_csv_file(sample_feedback_data):
    """Create a sample CSV file for end-to-end testing."""
    file_path = RAW_DATA_DIR / "e2e_test_feedback.csv"
    sample_feedback_data.to_csv(file_path, index=False)
    yield file_path
    file_path.unlink(missing_ok=True)

def test_end_to_end_workflow(sample_csv_file):
    """Test the complete workflow from data loading to analysis."""
    # 1. Test API health check
    health_response = client.get("/api/v1/health")
    assert health_response.status_code == 200
    assert health_response.json() == {"status": "healthy"}

    # 2. Test file upload
    with open(sample_csv_file, "rb") as f:
        upload_response = client.post(
            "/api/v1/upload",
            files={"file": ("e2e_test_feedback.csv", f, "text/csv")}
        )
    assert upload_response.status_code == 200
    assert "message" in upload_response.json()

    # 3. Test class-level analysis
    class_analysis = client.get(
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
    assert class_analysis.status_code == 200
    class_results = class_analysis.json()
    assert "statistics" in class_results
    assert "llm_analysis" in class_results
    assert "model_used" in class_results
    assert class_results["statistics"]["total_students"] == 4  # Unique students
    assert class_results["statistics"]["total_groups"] == 2

    # 4. Test group-level analysis
    group_analysis = client.get(
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
    assert group_analysis.status_code == 200
    group_results = group_analysis.json()
    assert "statistics" in group_results
    assert "llm_analysis" in group_results
    assert group_results["statistics"]["group_name"] == "Group A"
    assert group_results["statistics"]["student_count"] == 2  # Unique students in Group A

    # 5. Test student-level analysis
    student_analysis = client.get(
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
    assert student_analysis.status_code == 200
    student_results = student_analysis.json()
    assert "student_data" in student_results
    assert "llm_analysis" in student_results
    assert "evaluation_results" in student_results
    assert student_results["student_data"]["student_name"] == "Student 1"
    assert len(student_results["student_data"]["feedback"]) == 2  # Two feedback entries for Student 1

    # 6. Test feedback comparison
    comparison = client.get(
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
    assert comparison.status_code == 200
    comparison_results = comparison.json()
    assert "student_analysis" in comparison_results
    assert "comparison" in comparison_results
    assert "model_used" in comparison_results

def test_error_handling():
    """Test error handling in the API."""
    # Test with non-existent file
    response = client.get(
        "/api/v1/class-analysis",
        params={
            "files": ["nonexistent.csv"],
            "model_key": "llama-3.1-8b"
        }
    )
    assert response.status_code == 500

    # Test with invalid model key
    response = client.get(
        "/api/v1/class-analysis",
        params={
            "files": ["test.csv"],
            "model_key": "invalid-model"
        }
    )
    assert response.status_code == 500

    # Test with missing required parameters
    response = client.get("/api/v1/class-analysis")
    assert response.status_code == 422  # FastAPI validation error 