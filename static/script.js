document.addEventListener('DOMContentLoaded', () => {
    const videoUrlInput = document.getElementById('video-url');
    const questionInput = document.getElementById('question');
    const askButton = document.getElementById('ask-button');
    const loadingElement = document.getElementById('loading');
    const answerElement = document.getElementById('answer');
    const answerContent = document.getElementById('answer-content');
    const errorElement = document.getElementById('error');
    const errorContent = document.getElementById('error-content');

    function showLoading() {
        loadingElement.style.display = 'block';
        answerElement.style.display = 'none';
        errorElement.style.display = 'none';
        askButton.disabled = true;
    }

    function showAnswer(answer) {
        loadingElement.style.display = 'none';
        answerElement.style.display = 'block';
        errorElement.style.display = 'none';
        answerContent.textContent = answer;
        askButton.disabled = false;
    }

    function showError(error) {
        loadingElement.style.display = 'none';
        answerElement.style.display = 'none';
        errorElement.style.display = 'block';
        errorContent.textContent = error;
        askButton.disabled = false;
    }

    function validateInputs() {
        const videoUrl = videoUrlInput.value.trim();
        const question = questionInput.value.trim();
        
        if (!videoUrl || !question) {
            showError('Please enter both a YouTube URL and a question');
            return false;
        }
        
        // Basic YouTube URL validation
        if (!videoUrl.match(/^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+$/)) {
            showError('Please enter a valid YouTube URL');
            return false;
        }
        
        return true;
    }

    async function getAnswer() {
        if (!validateInputs()) return;

        showLoading();

        try {
            const response = await fetch('/api/answer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    video_url: videoUrlInput.value.trim(),
                    question: questionInput.value.trim()
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'An error occurred');
            }

            showAnswer(data.answer);
        } catch (error) {
            showError(error.message);
        }
    }

    askButton.addEventListener('click', getAnswer);

    // Allow Enter key to submit in question textarea
    questionInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            getAnswer();
        }
    });
}); 