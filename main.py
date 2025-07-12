from langchain_ollama import ChatOllama, OllamaEmbeddings
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
# Initialize model and embeddings
llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
    max_tokens=2048
)
embedding_fn = OllamaEmbeddings(model=MODEL_NAME) # or any ollama pulled embedding model

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
if docs:
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
    chat_history: List[tuple[str, str]]

#retrieval
def retrieve_node(state: RGraph):
    try:
        docs = retriever.invoke(state["question"])
        context_text = [doc.page_content for doc in docs]
        # print(f"Retrieved {len(docs)} documents: {context_text[:100]}...")  # log context for debugging
        return {"context": context_text, "question": state["question"], "metadata": state.get("metadata", {})}
    except Exception as e:
        print(f"Error in retrieve_node: {e}")
        return {"context": [], "question": state["question"], "metadata": state.get("metadata", {})}

def router(state: RGraph):
    context = state["context"]
    question = state["question"]
    if not context:
        print("No context retrieved, routing to fallback")
        return {"decision": "fallback", "context": context, "question": question, "metadata": state["metadata"]}


    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an expert in document relevance classification.
        Given the 'Context' and the 'Question', determine if the 'Context' contains enough relevant information to answer the question.
        Respond ONLY with 'yes' or 'no'. Consider partial or indirect relevance as sufficient for a 'yes' response."""),
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
    return {"decision": decision, "context": context, "question": question, "metadata": state["metadata"]}

def generation(state: RGraph):
    contexts = state["context"]
    question = state["question"]
    metadata = state["metadata"]
    context_str = "\n\n".join(contexts) if isinstance(contexts, list) else contexts
    # Hardcoded metadata :C
    if "title" in question.lower() and metadata.get("title"):
        context_str += f"\n\nDocument metadata title: {metadata.get('title')}"
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Use the provided 'Information' to answer the 'Question'.
         If the 'Information' is not sufficient, state that you don't have enough information but provide any relevant details available.
         Keep your answer concise and relevant."""),
        ("human", "'Question': {question}\n'Information': {context}")
    ])
    rag_chain = rag_prompt | llm | StrOutputParser()
    try:
        response = rag_chain.invoke({
            "question": question,
            "context": context_str
        })
        return {"generate": response, "context": contexts, "question": question, "metadata": metadata}
    except Exception as e:
        print(f"Error in generation: {e}")
        return {"generate": "Error generating response", "context": contexts, "question": question, "metadata": metadata}

def fallback(state: RGraph): # when context failed, rely on general knowledge
    question = state["question"]
    metadata = state["metadata"]
    fallback_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the question using your general knowledge, without relying on any document context. Provide a concise and accurate response."),
        ("human", "{question}")
    ])
    fallback_chain = fallback_prompt | llm | StrOutputParser()
    try:
        response = fallback_chain.invoke({"question": question})
        return {"generate": response, "context": state["context"], "question": question, "metadata": metadata}
    except Exception as e:
        print(f"Error in fallback: {e}")
        return {"generate": "Error generating response", "context": state["context"], "question": question, "metadata": metadata}

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
        "metadata": metadata  
    }
    try:
        final_state = None
        for state in app.stream(inputs):
            for node_name, node_state in state.items():
                # print(f"Processing node: {node_name}") #debugging
                final_state = node_state
        if final_state and "generate" in final_state and final_state["generate"]:
            print(f"Response: {final_state['generate']}")
        else:
            print("No response generated.")
    except Exception as e:
        print(f"An error occurred: {e}")