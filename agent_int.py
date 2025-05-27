import streamlit as st
from audiorecorder import audiorecorder # For recording audio
import speech_recognition as sr # For converting speech to text
import requests # For sending data to N8N
import io # For handling audio bytes
from pydub import AudioSegment # For audio format conversion
import json # To potentially help inspect response, though requests handles most JSON parsing
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

    #fetch the audio content
    audio_content = response.json()['audioContent']

    # Decode the base64 string
    audio_out = base64.b64decode(audio_content)

    return audio_out

def play_audio(audio):
    st.audio(audio, format='audio/mp3')


# --- Initialize Gemini Chat ---
if 'gemini_chat' not in st.session_state:
    # Replace with your actual Google API key
    st.session_state.gemini_chat = GeminiChat()

# --- Session State for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

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

# --- Audio Recorder ---
st.markdown("---")
st.subheader("Record your message:")
audio_bytes = audiorecorder("Click to Speak", "Recording... Click to Stop")

if audio_bytes:
    # Display the recorded audio
    audio_bytes = audio_bytes.export().read()
    st.audio(audio_bytes, format="audio/wav")

    # Convert audio_bytes to WAV for SpeechRecognition
    text = None  # Initialize text variable
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)  # Reset stream position

        with sr.AudioFile(wav_io) as source:
            audio_data = r.record(source)  # Read the entire audio file

        # Recognize speech using Google Web Speech API
        st.info("Processing speech...")
        text = r.recognize_google(audio_data)
        st.success(f"Recognized: {text}")

    except sr.UnknownValueError:
        st.warning("Google Web Speech API could not understand audio.")
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Web Speech API; {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during speech recognition: {e}")
        st.exception(e)

    # --- Send Recognized Text to Gemini and Handle Response ---
    if text:  # Only proceed if text was successfully recognized
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": text})
        # Re-display messages to include the new user message immediately
        with st.chat_message("user"):
            st.markdown(text)

        st.info("Sending text to Gemini...")

        try:
            # Send to Gemini
            with st.spinner("Getting response from Gemini..."):
                response = st.session_state.gemini_chat.send_message(text)

            st.success("Successfully received response from Fashion Assistant!")
            st.info("Fashion Agent says:")
            st.markdown(response['text'])

            # Add text response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response['text']})
            with st.chat_message("assistant"):
                st.markdown(response['text'])

            # Extract image URLs from the text response
            if response['image_urls']:
                st.markdown("### 👗 Recommended Looks:")

                # Create columns for image display
                cols = st.columns(min(3, len(response['image_urls'])))

                for idx, image_url in enumerate(response['image_urls']):
                    with cols[idx % 3]:
                        try:
                            st.image(image_url)
                        except Exception as e:
                            st.warning(f"Could not load image: {str(e)}")

            # Generate and play speech
            audio = get_speech(response['text'])
            with st.container():
                st.write("Fashion Agent says")
                play_audio(audio)

        except Exception as e:
            error_msg = f"Error communicating with Gemini: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.markdown(error_msg)

else:
    st.info("Click the button above to start recording your voice query.")

# Add a small footer or instruction
st.markdown("---")
st.caption("Ensure your microphone is enabled in your browser for this site.")