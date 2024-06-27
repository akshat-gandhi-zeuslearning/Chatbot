# import streamlit as st
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from langchain_community.retrievers import BM25Retriever
# from langchain.docstore.document import Document
# from langchain_openai import ChatOpenAI
# from langchain.retrievers.ensemble import EnsembleRetriever
# from dotenv import load_dotenv
# import os
# import tiktoken  # Token counting library

# # Load environment variables
# load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")

# # Set your OpenAI API Key
# os.environ["OPENAI_API_KEY"] = api_key

# # Tokenizer initialization
# tokenizer = tiktoken.get_encoding("cl100k_base")

# # Function to count tokens
# def count_tokens(text):
#     return len(tokenizer.encode(text))

# # Function to split text into chunks with increased chunk size
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=1000)  # Increased chunk size
#     chunks = text_splitter.split_text(text)
#     # Print token count for each chunk
#     for i, chunk in enumerate(chunks):
#         print(f"Chunk {i+1} tokens: {count_tokens(chunk)}")
#     return chunks

# # Function to create and save a vector store from text chunks
# def get_vector_store(text_chunks):
#     embeddings = OpenAIEmbeddings()
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")

# # Function to create a conversational chain with memory
# def get_conversational_chain():
#     # Updated prompt template
#     prompt_template = """
#     If the response can have a link to any webpage in the embeddings, include the clickable link with context.

#     Context:
#     {context}

#     Previous Questions and Answers:
#     {conversation_history}

#     Question:
#     {question}

#     Answer:
#     """
#     model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "conversation_history", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# # Function to handle user input and process the question
# def user_input(user_question, chunks):
#     embeddings = OpenAIEmbeddings()
#     vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

#     # Create list of Document objects
#     documents = [Document(page_content=chunk) for chunk in chunks]

#     # Create BM25 retriever from documents
#     keyword_retriever = BM25Retriever.from_documents(documents)
#     keyword_retriever.k = 5

#     # Create Ensemble Retriever with both BM25 and vector retriever
#     ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, keyword_retriever], weights=[0.5, 0.5])
    
#     # Retrieve relevant documents
#     docs_rel = ensemble_retriever.get_relevant_documents(user_question)

#     # Prepare conversation history
#     conversation_history = ""
#     max_history_length = 1000  # Reduced max history length
#     if "conversation_history" in st.session_state:
#         conversation_history = "".join([
#             f"User: {entry['question']}\nBot: {entry['answer']}\n"
#             for entry in st.session_state.conversation_history[-3:]  # Limit to the last 3 exchanges
#         ])[:max_history_length]  # Ensure the conversation history doesn't exceed the max length

#     # Trim context if necessary
#     context = "\n".join([doc.page_content for doc in docs_rel])
#     max_context_length = 3000  # Reduced max context length
#     context = context[:max_context_length]  # Ensure the context doesn't exceed the max length

#     # Count tokens for context and conversation history
#     context_tokens = count_tokens(context)
#     conversation_history_tokens = count_tokens(conversation_history)
#     question_tokens = count_tokens(user_question)
#     total_tokens = context_tokens + conversation_history_tokens + question_tokens
#     print(f"Context tokens: {context_tokens}")
#     print(f"Conversation history tokens: {conversation_history_tokens}")
#     print(f"Question tokens: {question_tokens}")
#     print(f"Total tokens (before adding prompt): {total_tokens}")

#     # Use conversational chain for answering question
#     chain = get_conversational_chain()
#     response = chain({
#         "input_documents": docs_rel,  # Provide input_documents key with relevant documents
#         "context": context,
#         "conversation_history": conversation_history,
#         "question": user_question
#     }, return_only_outputs=True)

#     # Store conversation history in session state
#     if "conversation_history" not in st.session_state:
#         st.session_state.conversation_history = []
#     st.session_state.conversation_history.append({
#         "question": user_question,
#         "answer": response["output_text"]
#     })

#     return response["output_text"]

# # Main function to run the Streamlit app
# def main():
#     st.set_page_config(page_title="Chat with PDF using GPT-3.5💁", layout="centered")
#     st.header("Chat with PDF using GPT-3.5💁")

#     user_question = st.text_input("Ask a Question from the PDF Files")

#     if user_question:
#         with st.spinner("Generating response..."):
#             chunks = st.session_state.get("text_chunks", [])
#             if chunks:
#                 response = user_input(user_question, chunks)
#                 st.write("Reply:", response)
#             else:
#                 st.warning("Please upload and process a text file first.")

#         if "conversation_history" in st.session_state:
#             st.subheader("Conversation History")
#             for i, entry in enumerate(st.session_state.conversation_history):
#                 st.write(f"Q{i+1}: {entry['question']}")
#                 st.write(f"A{i+1}: {entry['answer']}")

#     # Process the text file generated by Scrapy spider
#     text_file_path = "output/text_content.txt"
#     if os.path.exists(text_file_path):
#         with open(text_file_path, 'r', encoding='utf-8') as file:
#             raw_text = file.read()
#             text_chunks = get_text_chunks(raw_text)
#             st.session_state["text_chunks"] = text_chunks
#             get_vector_store(text_chunks)
#             st.success("Processing Complete!")

# if __name__ == "__main__":
#     main()

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document
from langchain.retrievers.ensemble import EnsembleRetriever
from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModel
import torch

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Set your OpenAI API Key
os.environ["OPENAI_API_KEY"] = api_key

# Load a Hugging Face tokenizer and model for embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Function to get embeddings from a Hugging Face model
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Function to split text into chunks with increased chunk size
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=1000)  # Adjusted chunk size
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save a vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = [get_embeddings(chunk) for chunk in text_chunks]
    vector_store = FAISS.from_embeddings(embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational chain with memory
def get_conversational_chain():
    # Updated prompt template
    prompt_template = """
    If the response can have a link to any webpage in the embeddings, include the clickable link with context.

    Context:
    {context}

    Previous Questions and Answers:
    {conversation_history}

    Question:
    {question}

    Answer:
    """
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "conversation_history", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and process the question
def user_input(user_question, chunks):
    embeddings = [get_embeddings(chunk) for chunk in chunks]
    vector_store = FAISS.from_embeddings(embeddings)
    vector_store.save_local("faiss_index")
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Create list of Document objects
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Create BM25 retriever from documents
    keyword_retriever = BM25Retriever.from_documents(documents)
    keyword_retriever.k = 5

    # Create Ensemble Retriever with both BM25 and vector retriever
    ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, keyword_retriever], weights=[0.5, 0.5])
    
    # Retrieve relevant documents
    docs_rel = ensemble_retriever.get_relevant_documents(user_question)

    # Prepare conversation history
    conversation_history = ""
    max_history_length = 1000  # Reduced max history length
    if "conversation_history" in st.session_state:
        conversation_history = "".join([
            f"User: {entry['question']}\nBot: {entry['answer']}\n"
            for entry in st.session_state.conversation_history[-3:]  # Limit to the last 3 exchanges
        ])[:max_history_length]  # Ensure the conversation history doesn't exceed the max length

    # Trim context if necessary
    context = "\n".join([doc.page_content for doc in docs_rel])
    max_context_length = 3000  # Adjusted max context length
    context = context[:max_context_length]  # Ensure the context doesn't exceed the max length

    # Use conversational chain for answering question
    chain = get_conversational_chain()
    response = chain({
        "input_documents": docs_rel,  # Provide input_documents key with relevant documents
        "context": context,
        "conversation_history": conversation_history,
        "question": user_question
    }, return_only_outputs=True)

    # Store conversation history in session state
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    st.session_state.conversation_history.append({
        "question": user_question,
        "answer": response["output_text"]
    })

    return response["output_text"]

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Chat with Documents", layout="centered")
    st.header("Chat with Documents using GPT-3.5💁")

    user_question = st.text_input("Ask a Question from the Documents")

    if user_question:
        with st.spinner("Generating response..."):
            chunks = st.session_state.get("text_chunks", [])
            if chunks:
                response = user_input(user_question, chunks)
                st.write("Reply:", response)
            else:
                st.warning("Please process the text files first.")

        if "conversation_history" in st.session_state:
            st.subheader("Conversation History")
            for i, entry in enumerate(st.session_state.conversation_history):
                st.write(f"Q{i+1}: {entry['question']}")
                st.write(f"A{i+1}: {entry['answer']}")

    # Process the text files initially
    with open("output/text_content.txt", "r", encoding="utf-8") as file:
        raw_text = file.read()

    text_chunks = get_text_chunks(raw_text)
    st.session_state["text_chunks"] = text_chunks
    get_vector_store(text_chunks)
    st.success("Processing Complete!")

if __name__ == "__main__":
    main()
