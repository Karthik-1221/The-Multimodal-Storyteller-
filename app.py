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

# --- 1. Configuration and Setup ---

# Set page configuration for a better layout
st.set_page_config(layout="wide", page_title="The Multimodal Storyteller", page_icon="🪶")

# Load environment variables from .env file for local development
load_dotenv()

# Function to load API keys securely
def load_api_keys():
    """
    Loads API keys securely. It first tries to load from Streamlit's secrets manager
    (for cloud deployment). If that fails (e.g., locally), it falls back to
    loading from a .env file.
    """
    try:
        # Try to load from Streamlit's secrets (for deployed app)
        google_api_key = st.secrets["GOOGLE_API_KEY"]
        hf_api_token = st.secrets["HF_API_TOKEN"]
    except (KeyError, st.errors.StreamlitAPIException):
        # If secrets aren't found, fall back to .env file (for local development)
        google_api_key = os.getenv("GOOGLE_API_KEY")
        hf_api_token = os.getenv("HF_API_TOKEN")
        
    return google_api_key, hf_api_token

GOOGLE_API_KEY, HF_API_TOKEN = load_api_keys()

# Configure the Gemini API
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.error("Google API Key not found. Please set it in your .env file or Streamlit secrets.")
    st.stop()

# Hugging Face API Configuration
HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# --- 2. AI and Helper Functions ---

# Function to generate the World Bible
def generate_world_bible(theme, archetype, contradiction):
    prompt = f"""
    You are a world-building AI. Create a 'World Bible' for a new story. This document should be a rich, one-page summary establishing the world's history, main conflicts, character motivations, and tone. It must be consistent and creative.

    Core Theme: {theme}
    Protagonist Archetype: {archetype}
    The World's Core Contradiction: {contradiction}

    Generate the World Bible based on these inputs.
    """
    with st.spinner("Generating the core of your universe..."):
        # MODEL NAME UPDATED to a newer, more available model
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)
        return response.text

# The "Master Prompt" function
def generate_story_chapter(story_context, world_bible, user_choice):
    # MODEL NAME UPDATED to a newer, more available model
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    generation_config = genai.types.GenerationConfig(temperature=0.9)
    
    prompt = f"""
    You are a multi-persona Storytelling Engine. Follow these steps precisely.
    The user's choice for the last chapter was: "{user_choice}".
    The full story context so far is: "{story_context}".
    The secret World Bible for this universe is: "{world_bible}".

    Step 1: Act as a Literary Artist. Your voice is poetic and unconventional. Use at least one surprising metaphor. Write a rich, descriptive paragraph expanding on the user's choice, revealing something new and strange. This is the main narrative.
    Step 2: Act as a Plot Theorist who values surprise. Based on the new paragraph you just wrote, generate three distinct, single-sentence plot choices for the user to choose from next. Two should be logical progressions. The third MUST be a 'Wildcard' that subverts expectations or connects to the world's core contradiction.
    Step 3: Act as an Art Director. Based on the paragraph from Step 1, write a concise, descriptive prompt for an AI image generator. The prompt should be comma-separated keywords focusing on characters, actions, settings, and style (e.g., 'a lone man on a dark dock, clutching his head, a ghostly crown glowing above him, digital art, cinematic lighting').
    Step 4: Format your entire response as a single, raw JSON object with NO markdown formatting, using these exact keys: "narrative_chapter", "next_choices", and "image_prompt".
    """
    
    with st.spinner("The Storyteller is weaving the next chapter..."):
        try:
            response = model.generate_content(prompt, generation_config=generation_config)
            cleaned_json_string = response.text.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned_json_string)
            return data
        except (json.JSONDecodeError, ValueError):
            st.error(f"Error decoding AI response. The AI returned an invalid format. Please try again.")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return None

# Function to generate an image from a prompt (with typo corrected)
def generate_image(prompt):
    if not HF_API_TOKEN:
        st.warning("Hugging Face API Token not found. Image generation is disabled.")
        return None
        
    with st.spinner("The AI artist is painting the scene..."):
        try:
            response = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": prompt})
            # CRITICAL FIX: Changed 'status__code' to 'status_code'
            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content))
            else:
                st.warning(f"Image generation failed (the free model might be loading). Please try again in a moment. Status: {response.status_code}")
                return None
        except Exception as e:
            st.error(f"An error occurred during image generation: {e}")
            return None

# The Narration Component
def text_to_speech_player(text):
    safe_text = text.replace("'", "\\'").replace('"', '\\"').replace("\n", " ").strip()
    components.html(f"""
        <script>
            const synth = window.speechSynthesis;
            if (synth.speaking) {{ synth.cancel(); }}
            const utterance = new SpeechSynthesisUtterance("{safe_text}");
            utterance.pitch = 1;
            utterance.rate = 0.9;
            synth.speak(utterance);
        </script>
    """, height=0)

# --- 3. Streamlit Application UI and Logic ---

st.title("The Multimodal Storyteller 🪶")
st.markdown("Co-create a unique saga with AI. Forge a world, make choices, and bring your story to life with generated art and audio.")

# Initialize session state variables
if 'world_bible' not in st.session_state: st.session_state.world_bible = None
if 'story_chapters' not in st.session_state: st.session_state.story_chapters = []
if 'latest_choices' not in st.session_state: st.session_state.latest_choices = []
if 'app_stage' not in st.session_state: st.session_state.app_stage = "world_forge"

# --- STAGE 1: World Forge ---
if st.session_state.app_stage == "world_forge":
    with st.form("world_forge_form"):
        st.header("Step 1: Forge Your World")
        theme = st.selectbox("Choose a Core Theme:", ["Revenge", "Discovery", "Betrayal", "Survival", "Redemption"])
        archetype = st.selectbox("Choose a Protagonist Archetype:", ["The Outcast", "The Reluctant Hero", "The Idealist", "The Trickster"])
        contradiction = st.text_input("What is a strange contradiction in this world?", "A city of high magic where everyone is profoundly bored.")
        
        if st.form_submit_button("Set the Stage"):
            st.session_state.world_bible = generate_world_bible(theme, archetype, contradiction)
            st.session_state.app_stage = "story_start"
            st.rerun()

# --- STAGE 2: Story Start ---
elif st.session_state.app_stage == "story_start":
    st.header("Step 2: Begin Your Saga")
    st.info("Your world has been created. Start your story with a single, compelling sentence.")
    
    with st.form("start_story_form"):
        initial_prompt = st.text_area("Your opening sentence:")
        if st.form_submit_button("Start the Saga") and initial_prompt:
            st.session_state.story_chapters.append({"text": initial_prompt, "image": None})
            st.session_state.latest_choices = [initial_prompt] 
            st.session_state.app_stage = "story_cycle"
            st.rerun()

# --- STAGE 3: Story Cycle ---
elif st.session_state.app_stage == "story_cycle":
    # The main creative loop
    st.header("Your Saga Unfolds...")
    for chapter in st.session_state.story_chapters:
        if chapter["image"]: st.image(chapter["image"], use_column_width=True)
        st.markdown(chapter["text"])
        st.markdown("---")
        
    # Narration Controls
    if st.session_state.story_chapters:
        full_story_text = " ".join([ch['text'] for ch in st.session_state.story_chapters])
        col1, col2, col3 = st.columns([2,1,1])
        if col1.button("🔊 Narrate Story"): text_to_speech_player(full_story_text)
        if col2.button("⏸️ Pause"): components.html("<script>window.speechSynthesis.pause();</script>", height=0)
        if col3.button("⏹️ Stop"): components.html("<script>window.speechSynthesis.cancel();</script>", height=0)

    st.header("What Happens Next?")
    
    # Generate the next chapter if a choice has been made
    if st.session_state.latest_choices:
        choice_made = st.radio("Choose a path:", st.session_state.latest_choices, key="choice_radio")
        if st.button("Weave Next Chapter"):
            story_so_far = " ".join([ch['text'] for ch in st.session_state.story_chapters])
            ai_response = generate_story_chapter(story_so_far, st.session_state.world_bible, choice_made)
            
            if ai_response:
                new_image = generate_image(ai_response["image_prompt"])
                st.session_state.story_chapters.append({"text": ai_response["narrative_chapter"], "image": new_image})
                st.session_state.latest_choices = ai_response["next_choices"]
                st.rerun()

# --- Footer and Restart Option ---
st.markdown("---")
if st.session_state.app_stage != "world_forge":
    if st.button("Start a New Saga (Restart)"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

st.markdown("""
<div style="text-align: center; padding: 10px;">
    <p>Created by <b>Karthik</b></p>
</div>
""", unsafe_allow_html=True)