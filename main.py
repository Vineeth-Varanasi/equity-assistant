import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
import shutil

from dotenv import load_dotenv
load_dotenv()

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

def load_url(url):
    try:
        # Headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Specific handling for MoneyControl articles
        if 'moneycontrol.com' in url:
            # Get the article content
            article_div = soup.find('div', {'class': 'content_wrapper'})
            if article_div:
                # Remove unwanted elements
                for div in article_div.find_all('div', {'class': ['attribution_block', 'related-news', 'ad', 'advertisement']}):
                    div.decompose()
                    
                # Get the article text
                text = article_div.get_text(separator=' ', strip=True)
            else:
                st.warning("Could not find article content. Falling back to general text extraction.")
                text = soup.get_text()
        else:
            # For non-MoneyControl URLs, get all text
            text = soup.get_text()
            
        # Clean up the text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Print first 500 characters to verify content
        st.write("Preview of extracted content:")
        st.write(text[:500] + "...")
        
        return Document(page_content=text, metadata={"source": url})
    except Exception as e:
        st.error(f"Error loading URL {url}: {str(e)}")
        return None

st.title("Equity research")

st.sidebar.title("Finance article URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:  # Only add non-empty URLs
        urls.append(url)

process_url_clicked = st.sidebar.button("Process data")
persist_directory = "db"

main_placeholder = st.empty()
llm = ChatGroq(temperature=0.9, max_tokens=500, model_name="mixtral-8x7b-32768")

if process_url_clicked and urls:
    try:
        main_placeholder.text("STATUS: Data is loading...")
        documents = []
        
        for url in urls:
            if url:
                st.write(f"Processing URL: {url}")
                doc = load_url(url)
                if doc:
                    documents.append(doc)
                    st.success(f"Successfully loaded content from {url}")
        
        if not documents:
            st.error("No data was loaded from the URLs. Please check if they are valid.")
            st.stop()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len
        )
        
        main_placeholder.text("STATUS: Text splitting...")
        docs = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        
        main_placeholder.text("STATUS: Vector embedding...")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vectorstore.persist()
        main_placeholder.text("Processing complete!")
        
        st.success(f"Successfully processed {len(documents)} URLs")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.stop()

query = st.text_input("Question: ")
if query and os.path.exists(persist_directory):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        st.header("Answer:")
        st.write(result["answer"])
    except Exception as e:
        st.error(f"An error occurred while processing your question: {str(e)}")
