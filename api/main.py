"""
FastAPI application for feedback analysis.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import pandas as pd
from pathlib import Path
import json
from src.analysis.analyzer import FeedbackAnalyzer
from src.config.settings import RAW_DATA_DIR
from src.models.llm import LLMInterface

app = FastAPI(
    title="Feedback Analysis API",
    description="API for analyzing student feedback using LLMs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Initialize analyzer with default model
analyzer = FeedbackAnalyzer()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the dashboard page."""
    try:
        # Get class-level analysis for the dashboard
        analysis = analyzer.analyze_class_feedback(
            [str(RAW_DATA_DIR / "e2e_test_feedback.csv")],
            group_col="group",
            student_col="student",
            feedback_col="feedback",
            date_col="date"
        )
        
        # Prepare data for charts
        feedback_distribution = {
            "labels": ["Positive", "Negative", "Neutral", "Mixed"],
            "data": [30, 20, 40, 10]  # Example data, replace with actual data
        }
        
        group_performance = {
            "labels": ["Group A", "Group B"],
            "data": [85, 75]  # Example data, replace with actual data
        }
        
        # Example insights
        insights = [
            {
                "title": "Overall Performance",
                "description": "The class shows strong engagement and participation."
            },
            {
                "title": "Areas for Improvement",
                "description": "Documentation quality needs attention across groups."
            }
        ]
        
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "stats": analysis["statistics"],
                "feedback_distribution": feedback_distribution,
                "group_performance": group_performance,
                "insights": insights
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "error": str(e)
            }
        )

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Serve the upload page."""
    return templates.TemplateResponse(
        "upload.html",
        {"request": request}
    )

@app.post("/upload")
async def upload_feedback(file: UploadFile = File(...), model: str = "llama-3.1-8b"):
    """Handle file upload and redirect to dashboard."""
    try:
        # Save the uploaded file
        file_path = RAW_DATA_DIR / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Redirect to dashboard
        return RedirectResponse(url="/", status_code=303)
    except Exception as e:
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": Request,
                "error": str(e)
            }
        )

@app.get("/api/v1/models")
async def get_available_models():
    """
    Get available model configurations.
    """
    return LLMInterface.get_available_models()

@app.get("/api/v1/class-analysis")
async def get_class_analysis(
    files: List[str],
    model_key: str = "llama-3.1-8b",
    group_col: str = "group",
    student_col: str = "student",
    feedback_col: str = "feedback",
    date_col: Optional[str] = None
):
    """
    Get class-level analysis of feedback data.
    """
    try:
        # Create new analyzer with specified model
        analyzer = FeedbackAnalyzer(model_key=model_key)
        analysis = analyzer.analyze_class_feedback(
            files,
            group_col,
            student_col,
            feedback_col,
            date_col
        )
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/group-analysis")
async def get_group_analysis(
    files: List[str],
    group_name: str,
    model_key: str = "llama-3.1-8b",
    group_col: str = "group",
    student_col: str = "student",
    feedback_col: str = "feedback",
    date_col: Optional[str] = None
):
    """
    Get group-level analysis of feedback data.
    """
    try:
        # Create new analyzer with specified model
        analyzer = FeedbackAnalyzer(model_key=model_key)
        analysis = analyzer.analyze_group_feedback(
            files,
            group_name,
            group_col,
            student_col,
            feedback_col,
            date_col
        )
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/student-analysis")
async def get_student_analysis(
    files: List[str],
    group_name: str,
    student_name: str,
    model_key: str = "llama-3.1-8b",
    framework: str = "ICAP",
    group_col: str = "group",
    student_col: str = "student",
    feedback_col: str = "feedback",
    date_col: Optional[str] = None
):
    """
    Get student-level analysis of feedback data.
    """
    try:
        # Create new analyzer with specified model
        analyzer = FeedbackAnalyzer(model_key=model_key)
        analysis = analyzer.analyze_student_feedback(
            files,
            group_name,
            student_name,
            framework,
            group_col,
            student_col,
            feedback_col,
            date_col
        )
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/compare-feedback")
async def compare_feedback(
    files: List[str],
    group_name: str,
    student_name: str,
    model_key: str = "llama-3.1-8b",
    group_col: str = "group",
    student_col: str = "student",
    feedback_col: str = "feedback"
):
    """
    Compare feedback quality across different sources.
    """
    try:
        # Create new analyzer with specified model
        analyzer = FeedbackAnalyzer(model_key=model_key)
        comparison = analyzer.compare_feedback_quality(
            files,
            group_name,
            student_name,
            group_col,
            student_col,
            feedback_col
        )
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"} 