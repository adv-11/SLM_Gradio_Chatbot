# modules/model_handler.py
import os
from huggingface_hub import InferenceClient
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceEndpoint
from .config import MODEL_MAPPING

def get_inference_client():
    """Initialize and return a HuggingFace Inference Client."""
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("HuggingFace API token not found")
    return InferenceClient(api_key=hf_token)

def get_model_response(prompt, conversation_history, model_name, params):
    """
    Generate a response from the selected model.
    
    Args:
        prompt: User's input message
        conversation_history: List of previous messages
        model_name: Name of the selected model
        params: Dictionary of model parameters
        
    Returns:
        str: Model's response
    """
    try:
        client = get_inference_client()
        
        # Format conversation history for API
        messages = []
        for conv in conversation_history:
            if conv[0]:  # User message
                messages.append({"role": "user", "content": conv[0]})
            if conv[1]:  # Assistant message
                messages.append({"role": "assistant", "content": conv[1]})
        
        # If current message not in history yet
        if not messages or messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": prompt})
            
        # Get model ID from mapping
        model_id = MODEL_MAPPING.get(model_name)
        if not model_id:
            return "Error: Model not found in configuration."
            
        # Stream the response
        stream = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=params.get("max_length", 2000),
            temperature=params.get("temperature", 0.01),
            top_p=params.get("top_p", 0.9),
            stream=True
        )
        
        # Collect chunks into full response
        full_response = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                
        return full_response
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

def get_qa_response(query, model_name, document_store, params):
    """
    Generate a document-based QA response using the document store.
    
    Args:
        query: User's question
        model_name: Name of the selected model
        document_store: FAISS vector store containing document embeddings
        params: Dictionary of model parameters
        
    Returns:
        str: QA response based on document content
    """
    try:
        # Get model ID from mapping
        model_id = MODEL_MAPPING.get(model_name)
        if not model_id:
            return "Error: Model not found in configuration."
        
        # Set up QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=HuggingFaceEndpoint(
                repo_id=model_id,
                max_length=params.get("max_length", 2000),
                temperature=params.get("temperature", 0.01),
                top_p=params.get("top_p", 0.9),
                top_k=5,
                huggingfacehub_api_token=os.getenv('HF_TOKEN'),
            ),
            chain_type="stuff",
            retriever=document_store.as_retriever()
        )
        
        # Run the query against the document
        response = qa_chain({"query": query})
        return response["result"]
    
    except Exception as e:
        return f"Error generating document-based response: {str(e)}"