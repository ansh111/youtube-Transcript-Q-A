import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from youtube_transcript_api import YouTubeTranscriptApi   
import re
from langchain.schema import Document
from dotenv import load_dotenv
from googletrans import Translator
import time

load_dotenv()
## load the GROQ API Key
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

groq_api_key=st.secrets["GROQ_API_KEY"]

llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama3-70b-8192")

prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    <context>
    Question:{input}

    """

)

transcript = ""

def extract_video_id(url):
    # Regular expression to match YouTube video ID
    pattern = r'(?:youtube\.com\/(?:v\/|watch\?v=|live\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
    
    # Search for the video ID in the URL
    match = re.search(pattern, url)
    if match:
        return match.group(1)  # Return the video ID
    else:
        return None  # Return None if no match is found
    
def get_video_transcript(video_id):
        # Fetch the transcript for the video
        transcript = YouTubeTranscriptApi.get_transcript(video_id,languages=['hi','en','pt'])

         # Convert the transcript to a string format (plain text)
        transcript_text = ""
        for entry in transcript:
            transcript_text += f"{entry['text']} "  # Combine all caption text with a space
        
        return transcript_text.strip()  # Remove any trailing spaces  

st.title("Document Q&A using Youtube Transcript")
generic_url = st.text_input("Enter the youtube url to fetch transcript")

def generate_transcript_embeddings():
     st.session_state.docs = [Document(page_content=transcript)]
     st.write(st.session_state.docs) ## Document Loading
     st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
     st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
     st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)
    

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.vectors = []
        st.session_state.embeddings=OpenAIEmbeddings()

    generate_transcript_embeddings()
        
            


if st.button("Fetch Transcript"):
   transcript = get_video_transcript(extract_video_id(generic_url))
   print(transcript)
   create_vector_embedding()
   st.write("Vector Database is ready")

user_prompt=st.text_input("Enter your query from the research paper")

final_response = ""
if st.button("Answer"):
    if  user_prompt:
        document_chain=create_stuff_documents_chain(llm,prompt)
        retriever=st.session_state.vectors.as_retriever()
        retrieval_chain=create_retrieval_chain(retriever,document_chain)

        start=time.process_time()
        response=retrieval_chain.invoke({'input':user_prompt})
        print(f"Response time :{time.process_time()-start}")
        final_response =  response['answer']
        st.write(final_response)
        transcript = ""

    ## With a streamlit expander
        with st.expander("Document similarity Search"):
            for i,doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('------------------------')
    else:
        st.write("please enter a question to get an answer")    







