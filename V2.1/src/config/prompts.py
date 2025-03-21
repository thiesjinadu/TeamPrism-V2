"""
LLM prompt templates for different analysis levels and tasks.
"""

CLASS_LEVEL_PROMPT = """
You are an expert educational analyst. Analyze the following feedback data from multiple student groups:

{feedback_data}

Please provide a comprehensive analysis addressing:
1. Overall class trends and patterns
2. Group performance comparisons
3. Topic-wise analysis of strengths and weaknesses
4. Self-assessment vs peer-assessment alignment
5. Groups requiring immediate attention
6. Groups that need more challenging tasks

Format your response as a structured JSON with the following sections:
{
    "overall_trends": {},
    "group_comparisons": {},
    "topic_analysis": {},
    "assessment_alignment": {},
    "attention_needed": {},
    "challenge_needed": {}
}
"""

GROUP_LEVEL_PROMPT = """
Analyze the following feedback data for a specific group:

{group_data}

Provide detailed insights about:
1. Group dynamics and collaboration
2. Individual contributions
3. Topic mastery levels
4. Areas of improvement
5. Notable achievements

Format your response as a structured JSON with the following sections:
{
    "group_dynamics": {},
    "contributions": {},
    "topic_mastery": {},
    "improvement_areas": {},
    "achievements": {}
}
"""

STUDENT_LEVEL_PROMPT = """
Analyze the following feedback for a specific student:

{student_data}

Evaluate the feedback using the {framework} framework and provide:
1. Structured feedback analysis
2. Key strengths
3. Areas for improvement
4. Development recommendations
5. Notable patterns or concerns

Format your response as a structured JSON with the following sections:
{
    "framework_analysis": {},
    "strengths": [],
    "improvement_areas": [],
    "recommendations": [],
    "patterns": {}
}
"""

EVALUATION_PROMPT = """
As an expert educational evaluator, assess the quality of the following feedback:

{feedback_text}

Consider the following criteria:
1. Specificity and clarity
2. Constructive nature
3. Actionability
4. Alignment with learning objectives
5. Evidence-based observations

Provide a score (0-100) and detailed justification.
Format your response as:
{
    "score": number,
    "justification": string,
    "criterion_scores": {
        "specificity": number,
        "constructiveness": number,
        "actionability": number,
        "alignment": number,
        "evidence": number
    }
}
""" 