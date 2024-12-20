import streamlit as st
from src.embedding import get_embedding
from src.index import VectorStore
import numpy as np
from PIL import Image

# Initialize the vector database
vector_db = VectorStore()

# Load existing index
vector_db.load('fashion_dataset/index')

st.title("Image Retrieval System with Reranking")

uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])
text_query = st.text_input("Or enter a textual description:")

num_results = min(100, st.number_input("### Number of results: ", value=0, placeholder="Type a number"))
num_results = st.slider("Number of results to display:", min_value=1, max_value=100, value=num_results)

if "selected_images" not in st.session_state:
    st.session_state["selected_images"] = []

def display_results_in_grid(results, columns=3, selectable=False):
    """Display images in a grid layout, optionally allowing user selection."""
    selected = []
    for i in range(0, len(results), columns):
        cols = st.columns(columns)
        for col, result in zip(cols, results[i:i + columns]):
            with col:
                st.image(result, use_container_width=True)
                if selectable:
                    if st.checkbox(f"Select", key=result):
                        selected.append(result)
    return selected

def rerank(selected_results, num_results=5):
    """
    Rerank the original results based on user-selected results.
    Computes similarity between selected results and the rest.
    """
    selected_embeddings = [get_embedding(Image.open(path), is_text=False) for path in selected_results]
    selected_mean_embedding = np.mean(selected_embeddings, axis=0)  # Aggregate selection
    
    # Rerank results based on similarity to the mean embedding
    reranked_results = vector_db.search(selected_mean_embedding, num_results)
    return reranked_results

results = []

# Rerank button at the top
if st.button("Rerank Based on Selection"):
    if st.session_state["selected_images"]:
        st.subheader("Reranked Results:")
        results = rerank(st.session_state["selected_images"], num_results)
        display_results_in_grid(results, columns=3)
    else:
        st.warning("No images selected for reranking.")

if uploaded_file:
    st.subheader("Uploaded Image:")
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_container_width=False)

    st.subheader("Search Results:")
    results = vector_db.search(get_embedding(image, is_text=False), num_results)
    selected = display_results_in_grid(results, columns=3, selectable=True)

    if selected:
        st.session_state["selected_images"].extend(selected)

if text_query:
    st.subheader(f"Search Results for Query: '{text_query}'")
    results = vector_db.search(get_embedding(text_query, is_text=True), num_results)
    selected = display_results_in_grid(results, columns=3, selectable=True)

    if selected:
        st.session_state["selected_images"].extend(selected)
