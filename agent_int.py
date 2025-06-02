import streamlit as st
from audiorecorder import audiorecorder  # For recording audio
import speech_recognition as sr  # For converting speech to text
import requests  # For sending data to N8N
import io  # For handling audio bytes
from pydub import AudioSegment  # For audio format conversion
import json  # To potentially help inspect response, though requests handles most JSON parsing
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from agent_lc import GeminiChat
import base64
import os
import time

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "tv-demo.json"

# --- Initialize Speech Recognizer ---
r = sr.Recognizer()

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Voice Interaction with Fashion Agent")

st.title("🎙️ Voice Interaction with Fashion Agent")
st.markdown("""
Record your voice query. The app will convert it to text, send it to a Gemini powered langchain agent
as a query parameter, and display the agent's response.
""")


def get_token():
    if ('token' not in st.session_state or
            'token_expiry' not in st.session_state or
            time.time() > st.session_state.token_expiry):
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["google_credentials"],
            #"tv-demo.json",
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        request = Request()
        credentials.refresh(request)

        st.session_state.token = credentials.token
        st.session_state.token_expiry = time.time() + 3480  # 58 minutes

    return st.session_state.token


def get_speech(text):
    payload = {
        "input": {
            "markup": text
        },
        "voice": {
            "languageCode": "en-IN",
            "name": "en-IN-Chirp3-HD-Zephyr",
            "voiceClone": {}
        },
        "audioConfig": {
            "audioEncoding": "LINEAR16"
        }
    }
    json_data = json.dumps(payload)
    token = get_token()
    response = requests.post(
        "https://texttospeech.googleapis.com/v1/text:synthesize",
        headers={
            "Authorization": f"Bearer {token}",
            "x-goog-user-project": "gen-lang-client-0263672353",  # Replace with your project ID
            "Content-Type": "application/json; charset=utf-8"
        },
        data=json_data
    )

    # fetch the audio content
    audio_content = response.json()['audioContent']

    # Decode the base64 string
    audio_out = base64.b64decode(audio_content)

    return audio_out


def play_audio(audio):
    st.audio(audio, format='audio/mp3')


def process_audio_and_chat(audio_bytes):
    """Process audio recording and handle chat interaction"""
    # Convert audio_bytes to WAV for SpeechRecognition
    text = None
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)

        with sr.AudioFile(wav_io) as source:
            audio_data = r.record(source)

        # Recognize speech using Google Web Speech API
        st.info("Processing speech...")
        text = r.recognize_google(audio_data)
        st.success(f"Recognized: {text}")

    except sr.UnknownValueError:
        st.warning("Google Web Speech API could not understand audio.")
        return
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Web Speech API; {e}")
        return
    except Exception as e:
        st.error(f"An unexpected error occurred during speech recognition: {e}")
        return

    # Send to Gemini if text was recognized
    if text:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": text})

        st.info("Sending text to Gemini...")

        try:
            # Send to Gemini
            with st.spinner("Getting response from Gemini..."):
                response = st.session_state.gemini_chat.send_message(text)

            st.success("Successfully received response from Fashion Assistant!")

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response['text']})

            # Store response details for display after rerun
            st.session_state.latest_response = {
                'text': response['text'],
                'image_urls': response.get('image_urls', [])
            }

            # Set flag to indicate we should process the response
            st.session_state.show_latest_response = True

            # Force rerun to refresh the UI
            st.rerun()

        except Exception as e:
            error_msg = f"Error communicating with Gemini: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})


# --- Initialize Gemini Chat ---
if 'gemini_chat' not in st.session_state:
    st.session_state.gemini_chat = GeminiChat()

# --- Session State for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize audio recording counter to force widget refresh
if "audio_key" not in st.session_state:
    st.session_state.audio_key = 0

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    st.info("Fashion Agent is configured and ready!")

    st.markdown("---")
    st.subheader("Chat Management")

    # Clear chat history button
    if st.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state.gemini_chat.clear_history()
        st.session_state.messages = []
        st.session_state.audio_key += 1  # Reset audio recorder
        if 'latest_response' in st.session_state:
            del st.session_state.latest_response
        if 'show_latest_response' in st.session_state:
            del st.session_state.show_latest_response
        st.success("Chat history cleared!")
        st.rerun()

    # View chat history from Gemini memory
    if st.button("📜 View Gemini Memory", type="secondary"):
        gemini_history = st.session_state.gemini_chat.get_history()
        if gemini_history:
            st.write("**Gemini Conversation Memory:**")
            for i, msg in enumerate(gemini_history):
                role = "🧑 User" if msg.__class__.__name__ == "HumanMessage" else "🤖 Gemini"
                st.write(f"**{role}:** {msg.content}")
                if i < len(gemini_history) - 1:
                    st.write("---")
        else:
            st.info("No conversation history in Gemini memory yet.")

    # Show conversation count
    gemini_msg_count = len(st.session_state.gemini_chat.get_history())
    st.metric("Messages in Memory", f"{gemini_msg_count} messages")

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display latest response if available
if st.session_state.get('show_latest_response', False) and 'latest_response' in st.session_state:
    response = st.session_state.latest_response

    st.info("Fashion Agent says:")
    st.markdown(response['text'])

    # Display images if available
    if response['image_urls']:
        st.markdown("### 👗 Recommended Looks:")
        cols = st.columns(min(3, len(response['image_urls'])))
        for idx, image_url in enumerate(response['image_urls']):
            with cols[idx % 3]:
                try:
                    st.image(image_url)
                except Exception as e:
                    st.warning(f"Could not load image: {str(e)}")

    # Generate and play speech
    try:
        audio = get_speech(response['text'])
        with st.container():
            st.write("🔊 Fashion Agent says:")
            play_audio(audio)
    except Exception as e:
        st.warning(f"Could not generate speech: {str(e)}")

    # Clear the flag after displaying
    st.session_state.show_latest_response = False

# --- Audio Recorder with Dynamic Key ---
st.markdown("---")
st.subheader("Record your message:")

# Use a dynamic key to refresh the audio recorder widget
audio_bytes = audiorecorder(
    "Click to Speak",
    "Recording... Click to Stop",
    key=f"audio_recorder_{st.session_state.audio_key}"
)

if audio_bytes:
    # Check if this is a new recording by comparing with previous
    current_audio_data = audio_bytes.export().read()

    # Store hash of current audio to detect new recordings
    import hashlib

    current_hash = hashlib.md5(current_audio_data).hexdigest()

    if st.session_state.get('last_audio_hash') != current_hash:
        st.session_state.last_audio_hash = current_hash

        # Display the recorded audio
        st.audio(current_audio_data, format="audio/wav")

        # Process the audio
        process_audio_and_chat(current_audio_data)

        # Increment key to refresh audio recorder for next recording
        st.session_state.audio_key += 1

else:
    st.info("Click the button above to start recording your voice query.")

# Add a small footer or instruction
st.markdown("---")
st.caption("Ensure your microphone is enabled in your browser for this site.")