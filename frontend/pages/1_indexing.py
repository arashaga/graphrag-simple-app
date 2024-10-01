import streamlit as st
import os
import time
from pathlib import Path
from dotenv import load_dotenv
import PyPDF2

import sys
from pathlib import Path

# Add the 'frontend' directory to sys.path
frontend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(frontend_dir))

from api.indexing import run_indexing  # Import the run_indexing function

st.set_page_config(page_title="Indexing", page_icon="📄")

st.title("📄 Indexing")

# Initialize session state variables
if 'indexing_in_progress' not in st.session_state:
    st.session_state['indexing_in_progress'] = False

if 'processed_indexes' not in st.session_state:
    st.session_state['processed_indexes'] = []

# File upload
uploaded_file = st.file_uploader("Upload a TXT or PDF file", type=['txt', 'pdf'])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()

    # Save uploaded file to 'input' directory
    input_dir = Path('../input')
    input_dir.mkdir(parents=True, exist_ok=True)

    if file_extension == 'pdf':
        # Read PDF and extract text
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Save extracted text to a TXT file
        txt_file_path = input_dir / (uploaded_file.name.replace('.pdf', '.txt'))
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        st.success(f"Extracted text from PDF and saved as {txt_file_path.name}")
    elif file_extension == 'txt':
        # Save the uploaded TXT file
        txt_file_path = input_dir / uploaded_file.name
        with open(txt_file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded TXT file saved as {txt_file_path.name}")
    else:
        st.error("Unsupported file type.")

    # Button to start indexing
    if st.button("Start Indexing") and not st.session_state['indexing_in_progress']:
        st.session_state['indexing_in_progress'] = True
        indexing_status_placeholder = st.empty()
        indexing_status_placeholder.info("Indexing in progress...")
        with st.spinner("Indexing started. Please wait..."):
            try:
                # Run the indexing process synchronously
                run_indexing()
                # Update the list of processed indexes
                st.session_state['processed_indexes'].append(txt_file_path.name)
                indexing_status_placeholder.success("Indexing complete.")
            except Exception as e:
                indexing_status_placeholder.error(f"Indexing failed: {e}")
            finally:
                st.session_state['indexing_in_progress'] = False

# Display list of processed indexes
if st.session_state['processed_indexes']:
    st.write("### Processed Indexes:")
    for idx in st.session_state['processed_indexes']:
        st.write(f"- {idx}")
