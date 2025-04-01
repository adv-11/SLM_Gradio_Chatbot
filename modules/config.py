# modules/config.py

# Mapping of display names to HuggingFace model IDs
MODEL_MAPPING = {
    'Llama 3.2 : 1B': "meta-llama/Llama-3.2-1B-Instruct",
    'Phi-3.5': "microsoft/Phi-3.5-mini-instruct",
    'Gemma 2 : 2B': "google/gemma-1.1-2b-it",
    'Gemma 3 : 27B': "google/gemma-3-27b-it",
    'DeepSeek-V3-0324': "deepseek-ai/DeepSeek-V3-0324",
    'DeepSeek-R1': "deepseek-ai/DeepSeek-R1"
}

# Default parameters for models
DEFAULT_PARAMETERS = {
    "temperature": 0.01,
    "top_p": 0.9,
    "max_length": 2000
}

# System prompts for document QA
QA_SYSTEM_PROMPT = """
Use the document to answer the questions.
If you don't know the answer, just say that you don't know. Do not generate other questions.
Use three sentences maximum and keep the answer as concise as possible.
"""