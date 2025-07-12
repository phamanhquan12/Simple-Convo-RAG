# Simple-ConvoRAG

Simple-ConvoRAG is a Retrieval Augmented Generation (RAG) system that combines external knowledge from PDF documents with the general knowledge of a language model. It intelligently routes questions based on the relevance of retrieved context.

## Features
- **Document Retrieval**: Extracts relevant information from PDF files.
- **Context Relevance Classification**: Determines if the retrieved context is sufficient to answer the question.
- **Fallback Mechanism**: Provides answers using the model's general knowledge when context is insufficient.
- **Adaptive Pipeline**: Routes questions dynamically between retrieval-based and general knowledge-based responses.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/phamanhquan12/Simple-Convo-RAG.git
   cd Simple-ConvoRAG
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the application:
   ```bash
   python main.py
   ```
2. Select a PDF file when prompted.
3. Enter your question in the terminal.

## Example
### Input:
```
Enter your question (or 'exit' to quit): What is the main topic of this document?
```
### Output:
```
Response: The document discusses the permissions granted by Google for reproducing tables and figures.
```

## Requirements
- Python 3.10+
- `langchain-ollama`
- `langchain-community`
- `langchain-core`
- `langchain-text-splitters`


## License
This project is licensed under the MIT License.
