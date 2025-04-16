from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled, VideoUnavailable
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import re
import os
import requests
import time
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

def check_video_availability(video_id: str) -> bool:
    """Check if a YouTube video is available and accessible."""
    try:
        response = requests.get(f"https://www.youtube.com/watch?v={video_id}", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_transcript(video_id: str, max_retries: int = 3) -> str:
    """Fetch transcript for a YouTube video with retry logic."""
    for attempt in range(max_retries):
        try:
            # First check if video is available
            if not check_video_availability(video_id):
                return "This video is unavailable or private. Please check if the video exists and is publicly accessible."

            # List of languages to try (ordered by preference)
            languages = [
                'en', 'en-US', 'en-GB',  # English variants
                'a.en',  # Auto-generated English
                'es', 'fr', 'de', 'it',  # European languages
                'pt', 'ru', 'ja', 'ko',  # More languages
                'zh', 'hi', 'ar', 'nl',  # Additional languages
                'tr', 'pl', 'sv', 'fi',  # More European languages
                'el', 'da', 'no', 'hu',  # Additional European languages
                'cs', 'ro', 'bg', 'th',  # More languages
                'vi', 'id', 'ms', 'he',  # Asian and Middle Eastern languages
                'ta', 'te', 'bn'         # Indian languages
            ]
            
            # First try to get a transcript in any of the supported languages
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
                return " ".join([segment['text'] for segment in transcript_list])
            except NoTranscriptFound:
                # If no transcript found in preferred languages, try to get any available transcript
                try:
                    # Get list of all available transcripts
                    available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
                    
                    if not available_transcripts:
                        return "No transcripts are available for this video. The video might not have captions enabled."
                    
                    # Get the first available transcript
                    transcript = available_transcripts.find_transcript(languages)
                    transcript_list = transcript.fetch()
                    return " ".join([segment.text for segment in transcript_list])
                    
                except NoTranscriptFound:
                    if attempt == max_retries - 1:
                        return "No transcript is available for this video in any supported language."
                except TranscriptsDisabled:
                    return "Captions are disabled for this video. Please enable captions on YouTube to use this feature."
                except VideoUnavailable:
                    return "This video is unavailable or private. Please check if the video exists and is publicly accessible."
                except Exception as e:
                    if attempt == max_retries - 1:
                        return f"Error fetching transcript: {str(e)}"
            
            # If we get here, wait before retrying
            time.sleep(1)
            
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Error: {str(e)}"
            time.sleep(1)
    
    return "Failed to fetch transcript after multiple attempts. Please try again later."

def answer_question(video_url: str, question: str) -> str:
    """Answer a question about a YouTube video's content."""
    try:
        # Extract video ID
        video_id = extract_video_id(video_url)
        
        # Get transcript
        transcript = get_transcript(video_id)
        
        # Check if we got an error message instead of a transcript
        if transcript.startswith(("No transcript", "Captions are", "This video is", "Error fetching", "Error:", "Failed to fetch")):
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
        The transcript might be in any language, but please answer in English.

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