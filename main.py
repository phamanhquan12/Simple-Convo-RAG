from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import sys
from tkinter import Tk, filedialog

MODEL_NAME = "llama3"  # Default model name
K = 10  # Increased to retrieve more documents (original was 5)
MAX_HISTORY = 5 #Max history entries

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
    max_tokens=2048
)
# embedding_fn = OllamaEmbeddings(model=MODEL_NAME) # or any ollama pulled embedding model
embedding_fn = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

root = Tk()
root.withdraw()  
file_path = filedialog.askopenfilename(
    title="Select a document",
    filetypes=[("PDF files", "*.pdf"), ("Text files", "*.txt"), ("CSV files", "*.csv")]
)
if not file_path:
    print("No file selected. Exiting.")
    sys.exit(1)


try:
    loader = PyPDFLoader(file_path, extract_images=False)
    docs = loader.load()
    # Extract metadata (e.g., title) if available
    metadata = docs[0].metadata if docs else {}
    # print(f"File loaded successfully. Metadata: {metadata}") #debugging
except Exception as e:
    print(f"Error loading file: {e}")
    sys.exit(1)

#small chunks + overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  
    chunk_overlap=150,  
    add_start_index=True # add index as metadata to each chunk -> useful towards retrieval and debugging
)
chunks = text_splitter.split_documents(docs)
if docs: # just hardcoding the first page
    first_page_chunk = Document(
        page_content=docs[0].page_content,
        metadata={"source": "first_page", "title": metadata.get("title", "")}
    )
    chunks.append(first_page_chunk)
vectorstore = FAISS.from_documents(chunks, embedding_fn)
retriever = vectorstore.as_retriever(search_kwargs={"k": K})

#state
class RGraph(TypedDict):
    question: str
    context: List[str]
    decision: str
    generate: str
    metadata: dict
    chat_history: List[tuple[str, str]] #(q,a)

#retrieval
def retrieve_node(state: RGraph):
    try:
        docs = retriever.invoke(state["question"])
        context_text = [doc.page_content for doc in docs]
        print(f"Retrieved {len(docs)} documents.")  # log debugging
        return {"context": context_text, "question": state["question"], "metadata": state.get("metadata", {}), "chat_history": state["chat_history" ]}
    except Exception as e:
        print(f"Error in retrieve_node: {e}")
        return {"context": [], "question": state["question"], "metadata": state.get("metadata", {}), "chat_history": state["chat_history"]}

def router(state: RGraph):
    context = state["context"]
    question = state["question"]
    if not context:
        print("No context retrieved, routing to fallback")
        return {"decision": "fallback", "context": context, "question": question, "metadata": state["metadata"], "chat_history": state["chat_history"]}


    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an expert in document relevance classification.
        Given the 'Context' and the 'Question', determine if the 'Context' contains enough relevant information to answer the question.
        Respond ONLY with 'yes' or 'no'. Consider partial matches, synonyms, or indirect references (e.g., 'model' for 'architecture', 'paper' for 'document') as sufficient for a 'yes' response."""),
        #prompt for decision making
        ("human", "'Question': {question}\n'Context': {context}")
    ])

    decision_chain = decision_prompt | llm | StrOutputParser()
    context_str = "\n\n".join(context) if isinstance(context, list) else context
    try:
        result = decision_chain.invoke({
            "question": question,
            "context": context_str
        }).strip().lower()
        # print(f"Decision made: {result}") #debug
        decision = "relevant" if result == "yes" else "fallback"
    except Exception as e:
        print(f"Error in router: {e}")
        decision = "fallback"
    return {"decision": decision, "context": context, "question": question, "metadata": state["metadata"], "chat_history": state["chat_history"]}

def generation(state: RGraph):
    contexts = state["context"]
    question = state["question"]
    metadata = state["metadata"]
    chat_history = state["chat_history"]
    context_str = "\n\n".join(contexts) if isinstance(contexts, list) else contexts
    # Hardcoded metadata :C
    if "title" in question.lower() and metadata.get("title"):
        context_str += f"\n\nDocument metadata title: {metadata.get('title')}"
    history_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in chat_history[-MAX_HISTORY:]]) if chat_history else ""
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Use the provided 'Information' and 'Chat History' to answer the 'Question'.
         For summary related questions, provide a comprehensive overview of the document's key points, including architecture, results, and comparisons if available.
         If the 'Information' is not sufficient, state that you don't have enough information but provide any relevant details available.
         If the 'Chat History' contains relevant information, use it to enhance your response.
         If the 'Chat History' is empty, do not mention it in your response."""),
        ("human", "'Question': {question}\n'Information': {context}\n'Chat History': {chat_history}"),
    ])
    rag_chain = rag_prompt | llm | StrOutputParser()
    try:
        response = rag_chain.invoke({
            "question": question,
            "context": context_str,
            "chat_history": history_str
        })
        return {"generate": response, "context": contexts, "question": question, "metadata": metadata, "chat_history": chat_history}
    except Exception as e:
        print(f"Error in generation: {e}")
        return {"generate": "Error generating response", "context": contexts, "question": question, "metadata": metadata, "chat_history": chat_history}

def fallback(state: RGraph): # when context failed, rely on general knowledge
    question = state["question"]
    metadata = state["metadata"]
    chat_history = state["chat_history"]
    history_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in chat_history[-MAX_HISTORY:]]) if chat_history else ""
    fallback_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a general knowledge assistant.
        Answer the question based on your general knowledge, without relying on provided context.
        Use the 'Chat History' to provide context-aware responses if relevant. Provide a concise and accurate response.
        If the 'Chat History' is empty, do not mention it in your response.
        If the question is not clear or lacks context, ask for clarification instead of providing a vague answer.
         """),
        ("human", "'Question' :{question}\n 'Chat History': {chat_history}")
    ])
    fallback_chain = fallback_prompt | llm | StrOutputParser()
    try:
        response = fallback_chain.invoke({"question": question, "chat_history": history_str})
        return {"generate": response, "context": state["context"], "question": question, "metadata": metadata, "chat_history": chat_history}
    except Exception as e:
        print(f"Error in fallback: {e}")
        return {"generate": "Error generating response", "context": state["context"], "question": question, "metadata": metadata, "chat_history": chat_history}

def route_decision(state: RGraph):
    decision = state["decision"]
    return "generate" if decision == "relevant" else "fallback"


graph = StateGraph(RGraph)
graph.add_node("retrieve", retrieve_node)
graph.add_node("router", router)
graph.add_node("generate", generation)
graph.add_node("fallback", fallback)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "router")
graph.add_conditional_edges(
    "router", route_decision,
    {"generate": "generate", "fallback": "fallback"}
)
graph.add_edge("generate", END)
graph.add_edge("fallback", END)

app = graph.compile()

chat_history = []

while True:
    question = input("Enter your question (or 'exit' to quit): ").strip()
    if question.lower() == "exit":
        print("Exiting the application.")
        break
    if not question:
        print("Please enter a valid question.")
        continue

    inputs: RGraph = {
        "question": question,
        "context": [],
        "decision": "",
        "generate": "",
        "metadata": metadata ,
        "chat_history": chat_history 
    }
    try:
        final_state = None
        for state in app.stream(inputs):
            for node_name, node_state in state.items():
                # print(f"Processing node: {node_name}") #debugging
                final_state = node_state
        if final_state and "generate" in final_state and final_state["generate"]:
            print(f"Response: {final_state['generate']}")
            chat_history.append((question, final_state["generate"]))
            chat_history = chat_history[-MAX_HISTORY:]  # Keep only the last MAX_HISTORY entries
        else:
            print("No response generated.")
    except Exception as e:
        print(f"An error occurred: {e}")