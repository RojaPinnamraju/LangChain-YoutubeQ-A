from pytube import YouTube
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import re
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    # Regular expression pattern for YouTube video ID
    pattern = r'(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})'
    match = re.search(pattern, url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)

def get_transcript(video_url: str) -> str:
    """Fetch transcript for a YouTube video using pytube."""
    try:
        # Create YouTube object
        yt = YouTube(video_url)
        
        # Check if video exists and is accessible
        try:
            yt.check_availability()
        except:
            return "This video is unavailable or private. Please check if the video exists and is publicly accessible."
        
        # Get captions
        captions = yt.captions
        
        if not captions:
            return "No captions are available for this video. The video might not have captions enabled."
        
        # Try to get English captions first
        try:
            caption = captions.get_by_language_code('en')
            if caption:
                return caption.generate_srt_captions()
        except:
            pass
        
        # If English not available, try to get any available captions
        try:
            # Get the first available caption
            caption = list(captions.all())[0]
            return caption.generate_srt_captions()
        except:
            return "No captions are available for this video in any supported language."
            
    except Exception as e:
        return f"Error: {str(e)}"

def answer_question(video_url: str, question: str) -> str:
    """Answer a question about a YouTube video's content."""
    try:
        # Get transcript
        transcript = get_transcript(video_url)
        
        # Check if we got an error message instead of a transcript
        if transcript.startswith(("No captions", "This video is", "Error:")):
            return transcript
        
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
        The transcript is in SRT format, but please ignore the timing information and focus on the text content.

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