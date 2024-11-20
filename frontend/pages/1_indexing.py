import streamlit as st
import os
from pathlib import Path
import PyPDF2
import sys
import concurrent.futures

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

if 'indexing_status_message' not in st.session_state:
    st.session_state['indexing_status_message'] = ''

if 'indexing_future' not in st.session_state:
    st.session_state['indexing_future'] = None

# Display the indexing status
if st.session_state['indexing_in_progress']:
    st.info("Indexing in progress...")
else:
    if st.session_state['indexing_status_message']:
        if "failed" in st.session_state['indexing_status_message'].lower():
            st.error(st.session_state['indexing_status_message'])
        else:
            st.success(st.session_state['indexing_status_message'])

# Display list of processed indexes
if st.session_state['processed_indexes']:
    st.write("### Processed Indexes:")
    for idx in st.session_state['processed_indexes']:
        st.write(f"- {idx}")

# Check if indexing has completed
if st.session_state['indexing_future'] is not None:
    if st.session_state['indexing_future'].done():
        try:
            result = st.session_state['indexing_future'].result()
            st.session_state['indexing_status_message'] = "Indexing complete."
            # Append the processed file to the list
            if 'last_processed_file' in st.session_state:
                st.session_state['processed_indexes'].append(st.session_state['last_processed_file'])
        except Exception as e:
            st.session_state['indexing_status_message'] = f"Indexing failed: {e}"
        finally:
            st.session_state['indexing_in_progress'] = False
            st.session_state['indexing_future'] = None
        st.experimental_rerun()
    else:
        # If the future is not done, refresh the page after a short delay
        st.experimental_rerun()

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

    # Store the last processed file name
    st.session_state['last_processed_file'] = txt_file_path.name

    # Disable the Start Indexing button if indexing is in progress
    start_indexing_button = st.button(
        "Start Indexing",
        disabled=st.session_state['indexing_in_progress']
    )

    if start_indexing_button and not st.session_state['indexing_in_progress']:
        st.session_state['indexing_in_progress'] = True
        st.session_state['indexing_status_message'] = "Indexing in progress..."

        # Run the indexing process in a separate process
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
        future = executor.submit(run_indexing)
        st.session_state['indexing_future'] = future
        st.experimental_rerun()
else:
    # No file uploaded, disable the Start Indexing button
    st.button(
        "Start Indexing",
        disabled=True
    )
