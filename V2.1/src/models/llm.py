"""
LLM integration module for feedback analysis.
"""
import json
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from ..config.settings import (
    MODEL_CONFIGS,
    DEFAULT_MODEL,
    LLM_API_KEY
)

class LLMInterface:
    def __init__(self, model_key: str = DEFAULT_MODEL):
        """
        Initialize the LLM interface with the specified model.
        
        Args:
            model_key: Key of the model configuration to use
        """
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Model {model_key} not found in configurations")
            
        self.model_config = MODEL_CONFIGS[model_key]
        self.model_name = self.model_config["name"]
        self.max_tokens = self.model_config["max_tokens"]
        self.temperature = self.model_config["temperature"]
        self.context_length = self.model_config["context_length"]
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=LLM_API_KEY
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=LLM_API_KEY,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate a response from the LLM based on the input prompt.
        
        Args:
            prompt: The input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response as string
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def analyze_class_level(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze feedback at the class level.
        
        Args:
            feedback_data: Dictionary containing feedback data
            
        Returns:
            Analysis results as dictionary
        """
        from ..config.prompts import CLASS_LEVEL_PROMPT
        
        prompt = CLASS_LEVEL_PROMPT.format(
            feedback_data=json.dumps(feedback_data, indent=2)
        )
        
        response = self.generate_response(prompt)
        return json.loads(response)

    def analyze_group_level(self, group_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze feedback at the group level.
        
        Args:
            group_data: Dictionary containing group feedback data
            
        Returns:
            Analysis results as dictionary
        """
        from ..config.prompts import GROUP_LEVEL_PROMPT
        
        prompt = GROUP_LEVEL_PROMPT.format(
            group_data=json.dumps(group_data, indent=2)
        )
        
        response = self.generate_response(prompt)
        return json.loads(response)

    def analyze_student_level(
        self,
        student_data: Dict[str, Any],
        framework: str = "ICAP"
    ) -> Dict[str, Any]:
        """
        Analyze feedback at the student level.
        
        Args:
            student_data: Dictionary containing student feedback data
            framework: Analysis framework to use (e.g., "ICAP")
            
        Returns:
            Analysis results as dictionary
        """
        from ..config.prompts import STUDENT_LEVEL_PROMPT
        
        prompt = STUDENT_LEVEL_PROMPT.format(
            student_data=json.dumps(student_data, indent=2),
            framework=framework
        )
        
        response = self.generate_response(prompt)
        return json.loads(response)

    def evaluate_feedback(self, feedback_text: str) -> Dict[str, Any]:
        """
        Evaluate the quality of feedback using the LLM.
        
        Args:
            feedback_text: The feedback text to evaluate
            
        Returns:
            Evaluation results as dictionary
        """
        from ..config.prompts import EVALUATION_PROMPT
        
        prompt = EVALUATION_PROMPT.format(feedback_text=feedback_text)
        
        response = self.generate_response(prompt)
        return json.loads(response)

    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get available model configurations.
        
        Returns:
            Dictionary of available model configurations
        """
        return MODEL_CONFIGS 