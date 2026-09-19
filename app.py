import streamlit as st
from transformers import pipeline

# Page Configuration
st.set_page_config(page_title="Story Generator", page_icon="📜", layout="centered")

st.title("📜 AI Story Generator")
st.write("Generate creative text using Hugging Face's `distilgpt2` model.")

# Load generator directly
with st.spinner("Loading model..."):
    generator = pipeline("text-generation", model="distilbert/distilgpt2")

# Sidebar options
st.sidebar.header("Generation Options")
max_new_tokens = st.sidebar.slider(
    "Max New Tokens", 
    min_value=20, 
    max_value=200, 
    value=100, 
    step=10,
    help="Controls the maximum length of the newly generated text."
)

num_return_sequences = st.sidebar.slider(
    "Number of Stories", 
    min_value=1, 
    max_value=3, 
    value=1
)

# User input form
with st.form("story_form"):
    prompt = st.text_area(
        "Enter your prompt:", 
        value="Once upon a time in a land far, far away",
        height=100  # 修正：将 rows=3 改为 height=100
    )
    submit_button = st.form_submit_button("Generate Story")

# Handle generation
if submit_button:
    if not prompt.strip():
        st.warning("Please enter a prompt to generate text.")
    else:
        with st.spinner("Generating story..."):
            results = generator(
                prompt,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
                pad_token_id=50256,
                truncation=True
            )

        st.subheader("Generated Output")
        for idx, result in enumerate(results, start=1):
            if num_return_sequences > 1:
                st.markdown(f"**Option {idx}:**")
            st.info(result["generated_text"])
