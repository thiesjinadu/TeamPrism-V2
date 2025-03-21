import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path
import os
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

def load_model():
    """Load the Llama 3 model and tokenizer"""
    try:
        model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Using Llama 3.1 8B model
        print(f"Loading model {model_name}...")
        
        # Get the Hugging Face token
        token = os.getenv('HUGGINGFACE_TOKEN')
        if not token:
            print("Error: HUGGINGFACE_TOKEN not found in .env file")
            sys.exit(1)
            
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            torch_dtype=torch.float16,  # Use half precision for memory efficiency
            device_map="auto"  # Automatically handle model placement
        )
        print("Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Please make sure you have the correct model name and sufficient memory.")
        sys.exit(1)

def analyze_feedback(feedback_text, model, tokenizer):
    """Analyze feedback text using the model"""
    try:
        prompt = f"""Analyze the following student feedback and provide a summary based on these criteria:
        1. Technical Skills
        2. Communication
        3. Team Collaboration

        Feedback: {feedback_text}

        Please provide a brief summary for each criterion."""

        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=200, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response
    except Exception as e:
        print(f"Error analyzing feedback: {str(e)}")
        return "Error analyzing feedback"

def process_feedback_file(input_file):
    """Process the feedback CSV file and generate summaries"""
    try:
        # Create output directory if it doesn't exist
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Read the CSV file
        print(f"Reading input file: {input_file}")
        df = pd.read_csv(input_file)
        
        # Validate required columns
        required_columns = ['student_id', 'feedback_text']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {', '.join(missing_columns)}")
            print("Please ensure your CSV file has the following columns: student_id, feedback_text")
            sys.exit(1)
        
        # Load the model
        model, tokenizer = load_model()
        
        # Process each student's feedback
        results = []
        total_students = len(df)
        
        print(f"\nProcessing feedback for {total_students} students...")
        for index, row in df.iterrows():
            student_id = row['student_id']
            feedback = row['feedback_text']
            
            print(f"\nProcessing student {student_id} ({index + 1}/{total_students})...")
            print(f"Feedback: {feedback[:100]}...")  # Print first 100 chars of feedback
            
            analysis = analyze_feedback(feedback, model, tokenizer)
            print(f"Analysis complete for student {student_id}")
            
            results.append({
                'student_id': student_id,
                'feedback_summary': analysis
            })
        
        # Save results to CSV
        output_file = output_dir / "feedback_analysis.csv"
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"\nAnalysis complete! Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    input_file = "raw_data/feedback.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        print("Please place your CSV file in the raw_data directory with the name 'feedback.csv'")
        print("The CSV should have columns: student_id, feedback_text")
        sys.exit(1)
    
    process_feedback_file(input_file) 