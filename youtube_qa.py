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

def get_video_id(youtube_url: str) -> Optional[str]:
    """
    Extract video ID from YouTube URL.
    
    Args:
        youtube_url (str): The YouTube URL
        
    Returns:
        Optional[str]: The video ID if found, None otherwise
    """
    match = re.search(r"(?:v=|youtu\.be/)([\w\-]+)", youtube_url)
    return match.group(1) if match else None

def fetch_transcript(video_url: str) -> str:
    """
    Fetch transcript for a YouTube video, trying multiple languages.
    
    Args:
        video_url (str): The YouTube video URL
        
    Returns:
        str: The transcript text or an error message
    """
    video_id = get_video_id(video_url)
    if not video_id:
        return "❌ Invalid YouTube URL."
    
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # List of languages to try, ordered by preference
        languages = [
            "en", "en-US", "en-GB",  # English variants
            "hi", "es", "fr", "de",  # Major languages
            "it", "pt", "ru", "ja",  # More languages
            "ko", "zh", "ar", "nl",  # Additional languages
            "tr", "pl", "sv", "fi",  # European languages
            "el", "da", "no", "hu",  # More European languages
            "cs", "ro", "bg", "th",  # Additional languages
            "vi", "id", "ms", "he",  # Asian and Middle Eastern languages
            "ta", "te", "bn"         # Indian languages
        ]
        
        # First try to find a transcript in any of the supported languages
        try:
            transcript = transcript_list.find_transcript(languages)
            fetched = transcript.fetch()
            return " ".join([item.text for item in fetched])
        except NoTranscriptFound:
            # If no transcript found in preferred languages, try to get any available transcript
            available_transcripts = transcript_list._transcripts
            if available_transcripts:
                # Get the first available transcript
                transcript = available_transcripts[0]
                fetched = transcript.fetch()
                return " ".join([item.text for item in fetched])
            else:
                return "❌ No transcript found for this video."
                
    except TranscriptsDisabled:
        return "❌ Transcripts are disabled for this video."
    except Exception as e:
        logger.error(f"Error fetching transcript: {str(e)}")
        return f"❌ Unexpected error: {str(e)}"

def answer_question(video_url: str, user_question: str) -> str:
    """
    Answer a question about a YouTube video's content.
    
    Args:
        video_url (str): The YouTube video URL
        user_question (str): The question to answer
        
    Returns:
        str: The answer or an error message
    """
    try:
        # Fetch and validate transcript
        transcript = fetch_transcript(video_url)
        if transcript.startswith("❌"):
            return transcript
            
        logger.info(f"Transcript length: {len(transcript)} characters")
        
        # Split transcript into chunks with better parameters
        splitter = CharacterTextSplitter(
            chunk_size=1000,  # Increased chunk size
            chunk_overlap=200,  # Increased overlap
            length_function=len,
            separator=" "  # Split on spaces to keep words together
        )
        
        docs = splitter.create_documents([transcript])
        logger.info(f"Number of document chunks: {len(docs)}")
        
        # Create vector store with better parameters
        vectorstore = FAISS.from_documents(
            docs, 
            embeddings,
            distance_strategy="COSINE"  # Use cosine similarity
        )
        
        # Initialize memory with better parameters
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create the chain with better parameters
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}  # Get top 4 most relevant chunks
            ),
            memory=memory,
            return_source_documents=True,
            verbose=True
        )
        
        # Enhanced prompt template
        enhanced_question = f"""
        Analyze the video transcript and answer this question: {user_question}
        
        Response format:
        1. State the key information directly
        2. Include specific details if available
        3. Keep the response concise and factual
        
        Do not include:
        - Apologies or qualifiers
        - Unnecessary explanations
        - Phrases like "I'm sorry" or "unfortunately"
        
        Focus only on the factual content from the transcript.
        """
        
        # Get the answer
        result = chain({"question": enhanced_question})
        answer = result.get("answer", "No answer available.")
        
        # Check if the answer is too generic or indicates lack of information
        if any(phrase in answer.lower() for phrase in ["not mentioned", "not explicitly", "don't know", "can't find"]):
            # Get a summary of the video content
            summary_prompt = """
            Summarize the main content of this transcript in 2-3 concise sentences.
            Focus on key topics and facts.
            """
            
            summary_result = chain({"question": summary_prompt})
            summary = summary_result.get("answer", "")
            
            return f"""
            Transcript content:
            {summary}
            
            Available topics:
            1. Ask about specific details
            2. Request more information
            3. Inquire about particular moments
            """
            
        return answer
        
    except Exception as e:
        logger.error(f"Error in answer_question: {str(e)}")
        return f"⚠️ Error during Q&A processing: {str(e)}"

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