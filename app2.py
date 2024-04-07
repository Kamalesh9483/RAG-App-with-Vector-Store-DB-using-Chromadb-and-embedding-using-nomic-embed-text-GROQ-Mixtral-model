from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import gradio as gr
from langchain_groq import ChatGroq
import streamlit as st
import os
from groq import Groq
import bs4

load_dotenv() 

groq_api_key = os.environ['GROQ_API_KEY']

# model
chat = ChatGroq(
api_key = groq_api_key,
model_name = "mixtral-8x7b-32768"
                )

def RAG(urls,input_text):
    urls_list = urls.split(",")
    # Document loader
    text_documents=[WebBaseLoader(url).load() for url in urls_list]
    docs_list=[item for sublist in text_documents for item in sublist]
    # Text Splitting
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    documents=text_splitter.split_documents(docs_list)
    # Vector DB storage of embeddings
    db=Chroma.from_documents(
    documents=documents,
    collection_name="rag-chroma",
    embedding=OllamaEmbeddings(model='nomic-embed-text'),
    
    )
    retriever=db.as_retriever()
    
    # prompt
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context and answer the result in bullet points:

    <context>
    {context}
    </context>

    Question: {input}""")
    

    document_chain = create_stuff_documents_chain(chat, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input":input_text})
    
    return response["answer"]

# Define Gradio interface
# iface = gr.Interface(fn=RAG,
#                      inputs=[gr.Textbox(label="Enter URLs separated by comma"), gr.Textbox(label="Enter Question you want to ask based on entered URL")],
#                      outputs="text",
#                      title="RAG App with Vector Store DB using Chromadb and embedding using nomic-embed-text, GROQ (Mixtral model) for inferencing on Html content",
#                      description="Enter URLs and a question to query the documents.")
# iface.launch()

def main():
    st.subheader("RAG App with Vector Store DB using Chromadb and embedding using nomic-embed-text, GROQ (Mixtral model) for inferencing on Html content")  
    urls=st.text_input("Enter website links")
    question=st.text_input("Enter the Question you want to ask based on URL links")
    submit=st.button("Submit")
    if submit:
        response=RAG(urls,question)  
        st.text_area("The Result based on provided URLs and Question is...",response)
        # st.write('Database loaded!!!')      
        # input_text=st.text_input("Ask questions based on the entered website link")
        # st.text_area(RAG(db,input_text))
    
    
    
if __name__=="__main__":
    main()