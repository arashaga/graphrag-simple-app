import streamlit as st
import httpx
import os

# Set the FastAPI backend URL
API_URL = os.getenv("API_URL", "http://localhost:8000/query")

st.set_page_config(page_title="GraphRAG Chatbot", page_icon="🤖")

st.title("GraphRAG Chatbot")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Place the search type selection in the sidebar
with st.sidebar:
    st.radio(
        "Select Search Type",
        ('local', 'global'),
        index=0,
        key='search_type'
    )

# Retrieve the current search type from session state
current_search_type = st.session_state.get("search_type", "local")

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            # Display the search type with the user's message
            st.markdown(f"**[{message.get('search_type', 'Local').capitalize()} Search]** {message['content']}")
        else:
            st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Type your message here..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(f"**[{current_search_type.capitalize()} Search]** {prompt}")
    # Append to session state
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "search_type": current_search_type
    })

    # Send the user's message to the backend
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Send request to the backend
        try:
            with httpx.Client(timeout=None) as client:
                # Include 'search_type' in the JSON payload
                payload = {
                    "query": prompt,
                    "search_type": current_search_type
                }
                with client.stream("POST", API_URL, json=payload) as response:
                    response.raise_for_status()
                    for chunk in response.iter_text():
                        full_response += chunk
                        # Update assistant's message
                        message_placeholder.markdown(full_response + "▌")
        except Exception as e:
            full_response = f"An error occurred: {e}"
            message_placeholder.markdown(full_response)

        # Finalize the assistant's message
        message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
