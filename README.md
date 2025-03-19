# Feedback Analysis System

A robust backend application for analyzing student feedback using LLMs and providing insights at multiple levels.

## Project Structure

```
feedback_analysis/
├── data/                   # Data storage directory
│   ├── raw/               # Raw CSV files
│   └── processed/         # Processed data
├── src/
│   ├── config/           # Configuration files
│   │   ├── prompts.py    # LLM prompts
│   │   └── settings.py   # System settings
│   ├── data/             # Data processing modules
│   │   ├── loader.py     # Data loading utilities
│   │   └── preprocessor.py # Data preprocessing
│   ├── models/           # Model-related code
│   │   ├── llm.py        # LLM integration
│   │   └── evaluator.py  # Evaluation metrics
│   ├── analysis/         # Analysis modules
│   │   ├── class_level.py    # Class-level analysis
│   │   ├── group_level.py    # Group-level analysis
│   │   └── student_level.py  # Student-level analysis
│   └── visualization/    # Visualization components
├── tests/               # Test files
├── notebooks/          # Jupyter notebooks
└── api/               # API endpoints
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
LLM_API_KEY=your_api_key
MODEL_NAME=your_model_name
```

4. Run the application:
```bash
uvicorn api.main:app --reload
```

## Features

- Multi-level feedback analysis (Class, Group, Student)
- LLM-powered insights generation
- Modular evaluation metrics
- Interactive visualizations
- Configurable analysis frameworks

## Usage

1. Place your CSV files in the `data/raw` directory
2. Run the analysis pipeline
3. Access results through the API or web interface

## Configuration

- Modify prompts in `src/config/prompts.py`
- Adjust analysis frameworks in respective analysis modules
- Update evaluation metrics in `src/models/evaluator.py`
