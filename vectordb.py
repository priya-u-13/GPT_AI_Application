from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
import openai
from langchain.vectorstores import  Pinecone
from  langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone


def add_docs_to_vectordb(path):
    PINECONE_API_KEY=''
    embeddings = OpenAIEmbeddings(openai_api_key="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")
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=""
    )
    loader = TextLoader(path)
    docs=loader.load()
    text_splitter = RecursiveCharacterTextSplitter( chunk_size=500, chunk_overlap=0)
    docs_text = text_splitter.split_documents(docs)
    iname=""
    similarity_search=Pinecone.from_texts([doc.page_content for doc in docs_text],embeddings,index_name=f'{iname}')
    return similarity_search





