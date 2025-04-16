from yt_dlp import YoutubeDL
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

def get_transcript(video_url: str, max_retries: int = 3) -> str:
    """Fetch transcript for a YouTube video using yt-dlp."""
    for attempt in range(max_retries):
        try:
            # First check if video is available
            video_id = extract_video_id(video_url)
            if not check_video_availability(video_id):
                return "This video is unavailable or private. Please check if the video exists and is publicly accessible."

            # Configure yt-dlp options
            ydl_opts = {
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en', 'a.en'],
                'skip_download': True,
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            }

            # Try to get transcript
            with YoutubeDL(ydl_opts) as ydl:
                try:
                    # Get video info
                    info = ydl.extract_info(video_url, download=False)
                    
                    # Check if subtitles are available
                    if 'subtitles' in info or 'automatic_captions' in info:
                        # Try to get English subtitles first
                        if 'subtitles' in info and 'en' in info['subtitles']:
                            subtitle_url = info['subtitles']['en'][0]['url']
                        # Then try auto-generated English
                        elif 'automatic_captions' in info and 'a.en' in info['automatic_captions']:
                            subtitle_url = info['automatic_captions']['a.en'][0]['url']
                        else:
                            # Try to get any available subtitle
                            available_subs = info.get('subtitles', {}) or info.get('automatic_captions', {})
                            if available_subs:
                                first_lang = list(available_subs.keys())[0]
                                subtitle_url = available_subs[first_lang][0]['url']
                            else:
                                return "No captions are available for this video. The video might not have captions enabled."
                        
                        # Download and parse the subtitle file
                        response = requests.get(subtitle_url)
                        if response.status_code == 200:
                            # Parse the subtitle content (assuming it's in a simple format)
                            lines = response.text.split('\n')
                            transcript = ' '.join([line.strip() for line in lines if line.strip() and not line.strip().isdigit()])
                            return transcript
                        else:
                            if attempt == max_retries - 1:
                                return "Failed to download captions. Please try again later."
                    else:
                        return "No captions are available for this video. The video might not have captions enabled."
                        
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

def get_video_info(video_url: str) -> dict:
    """Get video metadata (title, description) using yt-dlp."""
    try:
        ydl_opts = {
            'skip_download': True,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            return {
                'title': info.get('title', ''),
                'description': info.get('description', ''),
                'uploader': info.get('uploader', ''),
                'duration': info.get('duration', 0),
                'view_count': info.get('view_count', 0),
                'like_count': info.get('like_count', 0)
            }
    except Exception as e:
        return {'error': str(e)}

def answer_question(video_url: str, question: str) -> str:
    """Answer a question about a YouTube video using available metadata."""
    try:
        # Get video information
        video_info = get_video_info(video_url)
        
        if 'error' in video_info:
            return f"Error getting video information: {video_info['error']}"
        
        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv('OPENAI_API_BASE')
        )
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant that answers questions about YouTube videos.
        Use the following video information to answer the question. If the answer cannot be found in the information, say "I don't know".
        
        Video Title: {title}
        Channel: {uploader}
        Description: {description}
        Duration: {duration} seconds
        Views: {view_count}
        Likes: {like_count}

        Question: {question}

        Answer:
        """)
        
        # Create chain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Get answer
        response = chain.invoke({
            "title": video_info['title'],
            "uploader": video_info['uploader'],
            "description": video_info['description'],
            "duration": video_info['duration'],
            "view_count": video_info['view_count'],
            "like_count": video_info['like_count'],
            "question": question
        })
        
        return response['text']
        
    except Exception as e:
        return f"Error: {str(e)}" 