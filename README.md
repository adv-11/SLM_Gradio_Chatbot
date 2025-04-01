# Gradio Chatbot : HuggingFace SLMs

A modular Gradio-based application for interacting with various small language models through the Hugging Face API.

## Project Structure

```
slm-poc/
├── main.py                 # Main application entry point
├── modules/
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Configuration settings and constants
│   ├── document_processor.py  # Document handling and processing
│   └── model_handler.py    # Model interaction and response generation
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Features

- Interactive chat interface with multiple language model options
- Document processing (PDF, DOCX, TXT) for question answering
- Adjustable model parameters (temperature, top_p, max_length)
- Streaming responses for better user experience
- Docker support for easy deployment

## Setup and Running

### Local Development

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your HuggingFace API token:
   ```
   HF_TOKEN=hf_your_token_here
   ```
4. Run the application:
   ```
   python main.py
   ```

### Docker Deployment

1. Build the Docker image:
   ```
   docker build -t slm-poc .
   ```
2. Run the container:
   ```
   docker run -p 7860:7860 -e HF_TOKEN=hf_your_token_here slm-poc
   ```

## Usage

1. Access the web interface at http://localhost:7860
2. Enter your HuggingFace API token if not provided via environment variables
3. Select your preferred model and adjust parameters
4. Start chatting with the model
5. Optionally upload documents for document-based Q&A

## Supported Models

T2T Inference models provided by Hugging Face via the Inference API

## License

This project is licensed under the MIT License - see the LICENSE file for details.
