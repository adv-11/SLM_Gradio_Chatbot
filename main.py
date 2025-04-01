# main.py
import os
import gradio as gr
import tempfile
from dotenv import load_dotenv
from modules.document_processor import process_document
from modules.model_handler import get_model_response, get_qa_response
from modules.config import MODEL_MAPPING, DEFAULT_PARAMETERS

# Load environment variables
load_dotenv()

def setup_api_key(api_key=None):
    """Set up the HuggingFace API key from input or environment variables."""
    if api_key and api_key.startswith('hf_'):
        os.environ['HF_TOKEN'] = api_key
        return True, "API key set successfully! ✅"
    elif os.getenv('HF_TOKEN') and os.getenv('HF_TOKEN').startswith('hf_'):
        return True, "API key already available! ✅"
    else:
        return False, "Please enter a valid HuggingFace API key. ⚠️"

def create_chat_interface():
    """Create the main chat interface for the application."""
    with gr.Blocks(title="💬 Small Language Models - POC") as demo:
        # Application header
        gr.Markdown("# 💬 Small Language Models - POC")
        gr.Markdown("This chatbot uses various Language Models such as Llama 3.2, Gemma 2, Gemma 3, Phi 3.5, DeepSeek-V3, and DeepSeek-R1.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Sidebar configuration
                with gr.Group():
                    api_key_input = gr.Textbox(
                        label="HuggingFace API Token", 
                        placeholder="Enter your HF API token (hf_...)", 
                        type="password"
                    )
                    api_key_status = gr.Markdown("Please enter your API key.")
                    api_key_button = gr.Button("Set API Key")
                
                with gr.Group():
                    gr.Markdown("## Models and Parameters")
                    model_dropdown = gr.Dropdown(
                        choices=list(MODEL_MAPPING.keys()),
                        label="Select Model",
                        value=list(MODEL_MAPPING.keys())[0]
                    )
                    
                    temperature_slider = gr.Slider(
                        label="Temperature", 
                        minimum=0.01, 
                        maximum=1.0, 
                        value=DEFAULT_PARAMETERS["temperature"],
                        step=0.01
                    )
                    
                    top_p_slider = gr.Slider(
                        label="Top P", 
                        minimum=0.01, 
                        maximum=1.0, 
                        value=DEFAULT_PARAMETERS["top_p"],
                        step=0.01
                    )
                    
                    max_length_slider = gr.Slider(
                        label="Max Length", 
                        minimum=20, 
                        maximum=2040, 
                        value=DEFAULT_PARAMETERS["max_length"],
                        step=5
                    )
                    
                    clear_button = gr.Button("Clear Chat History")
                
                with gr.Group():
                    gr.Markdown("## Document Upload")
                    file_upload = gr.File(
                        label="Upload Document (PDF, DOCX, TXT)",
                        file_types=["pdf", "docx", "txt"]
                    )
                    upload_status = gr.Markdown("")

            with gr.Column(scale=2):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    bubble_full_width=False
                )
                msg = gr.Textbox(
                    label="Enter your message",
                    placeholder="Type your message here...",
                    show_label=False
                )
                
                # State variables to track conversation and document processing
                conversation_state = gr.State([])
                document_store = gr.State(None)
                api_key_state = gr.State(False)
        
        # Set up event handlers
        api_key_button.click(
            setup_api_key, 
            inputs=[api_key_input],
            outputs=[api_key_state, api_key_status]
        )
        
        file_upload.upload(
            process_document,
            inputs=[file_upload, api_key_state],
            outputs=[document_store, upload_status]
        )
        
        # Function to handle chat messages
        def respond(message, conversation, model_name, temp, top_p, max_len, doc_store, api_ready):
            if not api_ready:
                return conversation, conversation, "Please set a valid API key first. ⚠️"
            
            if not message.strip():
                return conversation, conversation, upload_status.value
            
            # Update conversation with user message
            conversation.append([message, None])
            yield conversation, conversation, upload_status.value
            
            # Generate response based on whether document is uploaded
            if doc_store is not None:
                response = get_qa_response(
                    message, 
                    model_name, 
                    doc_store, 
                    {"temperature": temp, "top_p": top_p, "max_length": max_len}
                )
            else:
                response = get_model_response(
                    message, 
                    conversation, 
                    model_name, 
                    {"temperature": temp, "top_p": top_p, "max_length": max_len}
                )
            
            # Update conversation with assistant response
            conversation[-1][1] = response
            yield conversation, conversation, upload_status.value
        
        # Function to clear chat history
        def clear_history():
            return [], gr.update(value="Chat history cleared.")
        
        # Connect events
        msg.submit(
            respond,
            [msg, conversation_state, model_dropdown, temperature_slider, top_p_slider, max_length_slider, document_store, api_key_state],
            [chatbot, conversation_state, upload_status]
        )
        
        clear_button.click(
            clear_history,
            outputs=[conversation_state, upload_status]
        )
        
    return demo

if __name__ == "__main__":
    # Create and launch the application
    app = create_chat_interface()
    app.launch(share=False)