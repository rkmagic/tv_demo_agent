import streamlit as st
from audiorecorder import audiorecorder # For recording audio
import speech_recognition as sr # For converting speech to text
import requests # For sending data to N8N
import io # For handling audio bytes
from pydub import AudioSegment # For audio format conversion
import json # To potentially help inspect response, though requests handles most JSON parsing
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import base64
import os
import time

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "tv-demo.json"

# --- N8N Webhook Configuration ---
DEFAULT_N8N_WEBHOOK_URL = st.secrets["N8N_URL"]

# --- Initialize Speech Recognizer ---
r = sr.Recognizer()

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Voice Interaction with N8N Agent")

st.title("🎙️ Voice Interaction with N8N Agent")
st.markdown("""
Record your voice query. The app will convert it to text, send it to your N8N webhook agent
as a query parameter, and display the agent's response.
""")

def get_token():
    if ('token' not in st.session_state or
            'token_expiry' not in st.session_state or
            time.time() > st.session_state.token_expiry):
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["google_credentials"],
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
    print(response.json())
    audio_content = response.json()['audioContent']

    # Decode the base64 string
    audio_out = base64.b64decode(audio_content)

    return audio_out

def play_audio(audio):
    st.audio(audio, format='audio/mp3')

# --- Session State for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "n8n_webhook_url" not in st.session_state:
    st.session_state.n8n_webhook_url = DEFAULT_N8N_WEBHOOK_URL

# --- Sidebar for N8N Webhook URL Configuration ---
with st.sidebar:
    st.header("Configuration")
    st.session_state.n8n_webhook_url = st.text_input(
        "N8N Webhook Base URL", # Changed label to clarify it's the base URL
        value=st.session_state.n8n_webhook_url
    )
    if st.session_state.n8n_webhook_url == DEFAULT_N8N_WEBHOOK_URL and "YOUR_N8N_TEST_WEBHOOK_URL_HERE" in DEFAULT_N8N_WEBHOOK_URL:
         st.warning("Please update the N8N Webhook URL!")
    elif not st.session_state.n8n_webhook_url.startswith("http"):
         st.warning("Webhook URL should start with http or https")


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
    # Note: This uses the original audio_bytes which is what st.audio expects
    audio_bytes = audio_bytes.export().read()
    st.audio(audio_bytes, format="audio/wav")

    # Convert audio_bytes (likely opus or webm from browser) to WAV for SpeechRecognition
    text = None # Initialize text variable
    try:
        # audiorecorder gives bytes. We need to know its format to convert.
        # It's often opus in a webm container. Let's try loading it directly.
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0) # Reset stream position

        with sr.AudioFile(wav_io) as source: # Use the BytesIO stream (wav_io)
            audio_data = r.record(source) # Read the entire audio file

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
        st.exception(e) # Display full traceback for other errors

    # --- Send Recognized Text to N8N Webhook and Handle Response ---
    if text: # Only proceed if text was successfully recognized
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": text})
        # Re-display messages to include the new user message immediately
        # (This is a common Streamlit pattern as the script reruns)
        with st.chat_message("user"):
             st.markdown(text)


        if st.session_state.n8n_webhook_url and st.session_state.n8n_webhook_url.startswith("http"):
            st.info("Sending text to N8N...")

            # Prepare the query parameters
            params = {"query": text}

            try:
                # Send POST request with parameters in the query string
                response = requests.post(st.session_state.n8n_webhook_url, params=params, timeout=20) # Increased timeout slightly

                if response.status_code == 200:
                    st.success("Successfully sent to N8N. Receiving response...")
                    try:
                        # Attempt to parse JSON response
                        response_data = response.json()
                        st.write("N8N Raw Response:", response_data) # Optional: show raw response

                        # Extract the 'output' key
                        if "output" in response_data:
                            n8n_output_text = response_data["output"]
                            st.info("N8N Agent says:")
                            st.markdown(n8n_output_text) # Display the agent's output

                            # Add N8N response to chat history
                            st.session_state.messages.append({"role": "assistant", "content": n8n_output_text})
                            # Re-display messages including the assistant's response
                            with st.chat_message("assistant"):
                                 st.markdown(n8n_output_text)

                            audio=get_speech(n8n_output_text)
                            with st.container():
                                st.write("Agent Output")
                                play_audio(audio)

                        else:
                            error_msg = "N8N response was successful but missing the 'output' key."
                            st.warning(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            with st.chat_message("assistant"):
                                st.markdown(error_msg)

                    except json.JSONDecodeError:
                        error_msg = f"Received status 200 from N8N but couldn't parse JSON response. Response text: {response.text[:200]}..."
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        with st.chat_message("assistant"):
                            st.markdown(error_msg)

                else:
                    error_msg = f"Error sending to N8N or unexpected status: {response.status_code} - {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": f"N8N Error: {response.status_code}"})
                    with st.chat_message("assistant"):
                        st.markdown(f"N8N Error: {response.status_code}")

            except requests.exceptions.RequestException as e:
                error_msg = f"N8N connection error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.markdown(error_msg)
        else:
            st.warning("N8N Webhook URL is not properly configured. Text not sent.")

else:
    st.info("Click the button above to start recording your voice query.")

# Add a small footer or instruction
st.markdown("---")
st.caption("Ensure your microphone is enabled in your browser for this site.")