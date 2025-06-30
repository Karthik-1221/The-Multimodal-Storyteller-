# app.py

import streamlit as st
import google.generativeai as genai
import requests
import json
from PIL import Image
import io
import os
from dotenv import load_dotenv
import streamlit.components.v1 as components
import time

# --- 1. Configuration and Setup ---

st.set_page_config(layout="wide", page_title="The Multimodal Storyteller", page_icon="🪶")
load_dotenv()

def load_api_keys():
    try:
        # Prioritize Streamlit secrets for deployment
        google_api_key = st.secrets.get("GOOGLE_API_KEY")
        hf_api_token = st.secrets.get("HF_API_TOKEN")
        if not hf_api_token: # If secrets exist but key is missing, try .env
            raise KeyError
    except (AttributeError, KeyError):
        # Fallback to .env for local development
        google_api_key = os.getenv("GOOGLE_API_KEY")
        hf_api_token = os.getenv("HF_API_TOKEN")
    return google_api_key, hf_api_token

GOOGLE_API_KEY, HF_API_TOKEN = load_api_keys()

# --- NEW: Let's try a different, highly reliable model as a backup ---
HF_API_URL_SDXL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HF_API_URL_SD15 = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
# We will use SD 1.5 for the hardcoded test as it's often faster to load
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# --- 2. AI and Helper Functions ---
def generate_image(prompt):
    # This function uses the main SDXL model
    api_url = HF_API_URL_SDXL
    
    # This check is crucial
    if not HF_API_TOKEN or "hf_" not in HF_API_TOKEN:
        st.session_state.last_error = "Hugging Face API Token is missing or invalid. Please check your Streamlit secrets."
        return None

    info_placeholder = st.info(f"🎨 Artist is warming up SDXL model... This can take up to 2 minutes.")
    try:
        response = requests.post(api_url, headers=HF_HEADERS, json={"inputs": prompt}, timeout=180)
        
        if response.status_code == 200:
            info_placeholder.success("✅ Scene painted!")
            time.sleep(1)
            info_placeholder.empty()
            return Image.open(io.BytesIO(response.content))
        else:
            st.session_state.last_error = f"API Error (SDXL). Status: {response.status_code}, Details: {response.text}"
            info_placeholder.empty()
            return None
            
    except Exception as e:
        st.session_state.last_error = f"A Python exception occurred: {e}"
        info_placeholder.empty()
        return None

def simple_image_test():
    # This function uses the faster SD 1.5 model for a quick, reliable test
    api_url = HF_API_URL_SD15
    prompt = "a red apple on a table, photorealistic"
    
    if not HF_API_TOKEN or "hf_" not in HF_API_TOKEN:
        st.session_state.last_error = "Hugging Face API Token is missing or invalid. The simple test cannot run."
        return None

    info_placeholder = st.info("🧪 Running simple test with SD 1.5 model...")
    try:
        response = requests.post(api_url, headers=HF_HEADERS, json={"inputs": prompt}, timeout=120)

        if response.status_code == 200:
            info_placeholder.success("✅ Simple test SUCCEEDED!")
            return Image.open(io.BytesIO(response.content))
        else:
            st.session_state.last_error = f"Simple Test API Error (SD 1.5). Status: {response.status_code}, Details: {response.text}"
            info_placeholder.empty()
            return None

    except Exception as e:
        st.session_state.last_error = f"Simple Test Python Exception: {e}"
        info_placeholder.empty()
        return None

# (Other functions like generate_world_bible, generate_story_chapter, text_to_speech_player remain the same)
def generate_world_bible(theme, archetype, contradiction):
    prompt = f"..." # Keeping it short for brevity
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = model.generate_content(prompt)
    return response.text

def generate_story_chapter(story_context, world_bible, user_choice):
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = f"..." # Keeping it short for brevity
    response = model.generate_content(prompt)
    cleaned_json_string = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned_json_string)

def text_to_speech_player(text):
    safe_text = json.dumps(text)
    components.html(f"""<script>...</script>""", height=0)


# --- 3. Streamlit Application UI and Logic ---

st.title("The Multimodal Storyteller 🪶")

# --- DEFINITIVE API KEY STATUS PANEL ---
st.header("🔑 API Key Status")
col1, col2 = st.columns(2)
with col1:
    if GOOGLE_API_KEY and "google" in GOOGLE_API_KEY:
        st.success("Google API Key: Loaded Successfully")
    else:
        st.error("Google API Key: NOT FOUND")
with col2:
    if HF_API_TOKEN and "hf_" in HF_API_TOKEN:
        st.success("Hugging Face Token: Loaded Successfully")
    else:
        st.error("Hugging Face Token: NOT FOUND or Invalid")
st.markdown("---")

# Initialize session state
if 'last_error' not in st.session_state: st.session_state.last_error = ""
if 'app_stage' not in st.session_state: st.session_state.app_stage = "debug" # Start at debug stage

# Persistent Error Log Display
if st.session_state.last_error:
    st.error(f"Last Error Logged: {st.session_state.last_error}")
    if st.button("Clear Error Log"):
        st.session_state.last_error = ""
        st.rerun()
    st.markdown("---")


# --- NEW DEBUG STAGE ---
if st.session_state.app_stage == "debug":
    st.header("🛠️ Admin Debug Panel")
    st.info("First, verify your API keys above. If the Hugging Face Token is loaded, click the button below.")
    
    if st.button("RUN SIMPLE IMAGE TEST (SD 1.5)"):
        image = simple_image_test()
        if image:
            st.image(image)
        else:
            st.warning("Simple test failed. Check the error log above.")
            
    if st.button("Proceed to World Forge"):
        st.session_state.app_stage = "world_forge"
        st.rerun()

# --- WORLD FORGE STAGE ---
elif st.session_state.app_stage == "world_forge":
    # (Your world forge code remains the same)
    with st.form("world_forge_form"):
        st.header("Step 1: Forge Your World")
        theme = st.selectbox("Choose a Core Theme:", ["Revenge", "Discovery"])
        archetype = st.selectbox("Choose a Protagonist Archetype:", ["The Outcast", "The Reluctant Hero"])
        contradiction = st.text_input("Contradiction?", "A bored magic city.")
        if st.form_submit_button("Set the Stage"):
            # Mocking this to speed up debugging
            st.session_state.world_bible = "A world bible created."
            st.session_state.app_stage = "story_start"
            st.rerun()

# --- STORY START STAGE ---
elif st.session_state.app_stage == "story_start":
    # (Your story start code remains the same)
    st.header("Step 2: Begin Your Saga")
    with st.form("start_story_form"):
        initial_prompt = st.text_area("Your opening sentence:")
        if st.form_submit_button("Start the Saga") and initial_prompt:
            st.session_state.story_chapters = [{"text": initial_prompt, "image": None}]
            st.session_state.latest_choices = [initial_prompt]
            st.session_state.app_stage = "story_cycle"
            st.rerun()

# --- STORY CYCLE STAGE ---
elif st.session_state.app_stage == "story_cycle":
    st.header("Your Saga Unfolds...")
    # (Display logic remains the same)
    for chapter in st.session_state.story_chapters:
        if chapter.get("image"): st.image(chapter["image"])
        st.markdown(chapter["text"])
        st.markdown("---")

    st.header("What Happens Next?")
    if st.session_state.latest_choices:
        choice_made = st.radio("Choose a path:", st.session_state.latest_choices, key="choice_radio")
        if st.button("Weave Next Chapter"):
            # Mocking the AI response to focus only on the image generation
            story_so_far = "..."
            ai_response = {"narrative_chapter": "A new chapter unfolds.", "next_choices": ["Choice A", "Choice B"], "image_prompt": "cinematic photo of a castle on a hill at sunset"}
            
            if ai_response:
                st.info(f"DEBUG: Attempting to generate image with prompt: '{ai_response['image_prompt']}'")
                new_image = generate_image(ai_response["image_prompt"])
                
                # We need to check if new_image is valid before appending
                if new_image:
                    st.session_state.story_chapters.append({"text": ai_response["narrative_chapter"], "image": new_image})
                    st.session_state.latest_choices = ai_response["next_choices"]
                    st.rerun()
                else:
                    st.warning("Image generation failed. Check error log at the top of the page.")
