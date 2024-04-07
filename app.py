from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
# import chainlit as cl
from langchain_groq import ChatGroq
import os
from groq import Groq


load_dotenv() 

groq_api_key = os.environ['GROQ_API_KEY']

llm_groq = ChatGroq(
            groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768",
                         temperature=0.2)

# model_local=ChatOllama(model="mistral")

#Split data into chunks
urls=[
    "https://kamalesh.net/"
    
]

docs=[WebBaseLoader(url).load() for url in urls]
docs_list=[item for sublist in docs for item in sublist]
text_splitter=CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500,chunk_overlap=100)
doc_splits=text_splitter.split_documents(docs_list)

# convert documents to Embeddings and store them in Vector DB
vectorstore=Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OllamaEmbeddings(model='nomic-embed-text'),
    
)

# Retreiver
retriever=vectorstore.as_retriever()

# Initialize message history for conversation
message_history = ChatMessageHistory()

# Memory for conversational context
memory = ConversationBufferMemory(
    memory_key="chat_history",
    output_key="answer",
    chat_memory=message_history,
    return_messages=True,
    )

# Create a chain that uses the Chroma vector store
chain = ConversationalRetrievalChain.from_llm(
    llm=llm_groq,
    chain_type="stuff",
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
)
# # Before RAG
# print("Before RAG\n")
# before_rag_template="Summarize about {topic}"
# before_rag_prompt=ChatPromptTemplate.from_template(before_rag_template)
# before_rag_chain=before_rag_prompt|model_local|StrOutputParser()
# print(before_rag_chain.invoke({"topic": "India"}))

# # After RAG
# print("\n #########\ After RAG\n")
# after_rag_template=""" Answer the Question based only on following context:
# {context}
# Question: {question}
# """
# after_rag_prompt=ChatPromptTemplate.from_template(after_rag_template)
# after_rag_chain=(
#     {"context":retriever, "Question":RunnablePassthrough()}
#     | after_rag_prompt
#     | model_local
#     | StrOutputParser()
# )
# print(after_rag_chain.invoke("Who is Kamalesh"))



