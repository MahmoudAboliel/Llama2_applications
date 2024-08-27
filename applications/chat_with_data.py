from langchain_community.llms import Replicate
from typing import Any
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredEPubLoader
)
import gradio as gr
import logging
import pathlib
from langchain.schema import Document

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.schema import Document, BaseRetriever

from langchain.chains import ConversationalRetrievalChain
from langchain.chains.base import Chain
from langchain.memory import ConversationBufferMemory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import LLMChain


#===========================================SET API===========================================#
import dotenv
dotenv.load_dotenv()

#===========================================INIT DOCUMENT LOADER===========================================#
class EpubReader(UnstructuredEPubLoader):
    def __init__(self, file_path: str | list[str], ** kwargs: Any):
        super().__init__(file_path, **kwargs, mode="elements", strategy="fast")


class DocumentLoaderException(Exception):
    pass

class DocumentLoader(object):
    """Loads in a document with a supported extension."""
    supported_extentions = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".epub": EpubReader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader
    }

def load_document(temp_filepath: str) -> list[Document]:
    """Load a file and return it as a list of documents."""
    ext = pathlib.Path(temp_filepath).suffix
    loader = DocumentLoader.supported_extentions.get(ext)
    if not loader:
        raise DocumentLoaderException(
            f"Invalid extension type {ext}, cannot load this type of file"
        )
    loader = loader(temp_filepath)
    docs = loader.load()
    logging.info(docs)
    return docs

# Vector storage
#===========================================EXTRACT CONTENT & STORE===========================================#
def configure_retriever(docs: list[Document]) -> BaseRetriever:
    """Retriever to use."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap = 200)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    return vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

#===========================================INIT MODEL===========================================#
def configure_chain(retriever: BaseRetriever) -> Chain:
    """Configure chain with a retriever."""
    # Setup memory for contextual conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # Setup LLM and QA chain; set temperature low to keep hallucinations in check
    llm = Replicate(
        model="meta/meta-llama-3-8b-instruct",
        model_kwargs={"temperature": 0.2, "max_length": 500, "top_p": 1},
    )
    # Passing in a max_tokens_limit amount automatically
    # truncates the tokens when prompting your llm!

    # SET PROMPTING
    system_prompt = "You are Nemo, a helpful and respectful assistant designed to aid English learning at the {level} level. \
            If the user makes any mistakes in the language, rephrase his message correctly and then respond to it. \
            Your focus is on interactive, motivating conversations. \
            Keep responses concise and engaging, encouraging user interaction. \
            Aim to keep answers around 20 words whenever possible. \
            If the question requires external information, use Wikipedia to search. \
            If you can answer directly, use the LLM model."
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
        verbose=True,
        max_tokens_limit=4000)

import os
import tempfile
def configure_qa_chain(uploaded_files):
    """Read documents, configure retriever, and the chain."""
    # docs = []
    # temp_dir = tempfile.TemporaryDirectory()
    # for file in uploaded_files:
    #     temp_filepath = os.path.join(temp_dir.name, file)
    #     with open(temp_filepath, "wb") as f:
    #         f.write(file)
    #     docs.extend(load_document(temp_filepath))
    docs = load_document(uploaded_files)
    retriever = configure_retriever(docs=docs)
    return configure_chain(retriever=retriever)

#===========================================FINAL FUNCTION===========================================#
def generate_RAG_response(prompt, chat_with_file):
    # qa_chain = configure_qa_chain(file)
    response = chat_with_file.invoke(prompt)['answer']
    return response

file = configure_qa_chain('generative_AI.pdf')
while True:
    prompt = input("question: ")
    response = generate_RAG_response(prompt, file)
    print(f"response: {response}")