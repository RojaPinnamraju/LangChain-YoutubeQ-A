import os
import re
import logging
from typing import Optional, List
import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get OpenRouter credentials from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

if not OPENAI_API_KEY or not OPENAI_API_BASE:
    raise ValueError("Please set OPENAI_API_KEY and OPENAI_API_BASE environment variables")

# Initialize LLM with better parameters
llm = ChatOpenAI(
    model="openai/gpt-3.5-turbo",
    temperature=0.2,
    max_tokens=2500,
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_API_BASE
)

# Use HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    # Regular expression pattern for YouTube video ID
    pattern = r'(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})'
    match = re.search(pattern, url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)

def get_transcript(video_id: str) -> str:
    """Fetch transcript for a YouTube video."""
    try:
        # Try to get transcript in English first
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
    except:
        try:
            # If English fails, try to get any available transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        except Exception as e:
            raise Exception(f"Could not fetch transcript: {str(e)}")
    
    # Combine all transcript segments
    return " ".join([segment['text'] for segment in transcript_list])

def answer_question(video_url: str, question: str) -> str:
    """Answer a question about a YouTube video's content."""
    try:
        # Extract video ID
        video_id = extract_video_id(video_url)
        
        # Get transcript
        transcript = get_transcript(video_id)
        
        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_API_BASE')
        )
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant that answers questions about YouTube video content.
        Use the following transcript to answer the question. If the answer cannot be found in the transcript, say "I don't know".

        Transcript: {transcript}

        Question: {question}

        Answer:
        """)
        
        # Create chain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Get answer
        response = chain.invoke({
            "transcript": transcript,
            "question": question
        })
        
        return response['text']
        
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio UI
iface = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Textbox(label="YouTube Video URL", placeholder="Enter YouTube URL here..."),
        gr.Textbox(label="Ask a Question", placeholder="What would you like to know about this video?")
    ],
    outputs="text",
    title="🎥 YouTube Transcript Q&A",
    description="Ask questions about the content of any YouTube video!",
    examples=[
        ["https://www.youtube.com/watch?v=dQw4w9WgXcQ", "What is the main topic of this video?"],
        ["https://www.youtube.com/watch?v=dQw4w9WgXcQ", "Summarize the key points"]
    ]
)

if __name__ == "__main__":
    iface.launch() 