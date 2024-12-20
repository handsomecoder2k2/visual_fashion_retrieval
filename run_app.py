# Required imports
import streamlit as st
from src.embedding import get_embedding
from src.index import VectorStore
import numpy as np
from PIL import Image
from functools import lru_cache
import concurrent.futures
from typing import List, Tuple
import io
import base64

# Cache embeddings to speed up repeat searches
@st.cache_data(ttl=3600)
def get_cached_embedding(image_bytes: bytes, is_text: bool = False) -> np.ndarray:
    if is_text:
        return get_embedding(image_bytes.decode(), is_text=True)
    image = Image.open(io.BytesIO(image_bytes))
    return get_embedding(image, is_text=False)

# Resize images to save memory and improve loading speed
@lru_cache(maxsize=100)
def load_and_resize_image(image_path: str, max_size: Tuple[int, int] = (300, 300)) -> bytes:
    try:
        with Image.open(image_path) as img:
            img.thumbnail(max_size)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format=img.format or 'JPEG', quality=85)
            return img_byte_arr.getvalue()
    except Exception as e:
        st.error(f"Error loading image {image_path}: {e}")
        return None

# Process multiple images at once using threads
def batch_process_images(image_paths: List[str], process_func, batch_size: int = 4) -> List:
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        future_to_path = {executor.submit(process_func, path): path for path in image_paths}
        for future in concurrent.futures.as_completed(future_to_path):
            result = future.result()
            if result is not None:
                results.append(result)
    return results

# Keep track of which images are selected
def update_selection(image_path: str, is_selected: bool):
    if "selected_images" not in st.session_state:
        st.session_state["selected_images"] = set()
    
    if is_selected:
        st.session_state["selected_images"].add(image_path)
    else:
        st.session_state["selected_images"].discard(image_path)

# Cache search results to avoid recomputing
@st.cache_data(ttl=300)
def cached_search(embedding_bytes: bytes, num_results: int) -> List[str]:
    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
    results = vector_db.search(embedding, num_results)
    return list(results) if isinstance(results, set) else results

# Handle both image and text searches
def handle_search(vector_db, image=None, text=None, num_results=5):
    try:
        if image:
            if isinstance(image, (bytes, io.BytesIO)):
                pil_image = Image.open(image).convert('RGB')
            else:
                pil_image = image.convert('RGB')
            embedding = get_embedding(pil_image, is_text=False)
        elif text:
            embedding = get_embedding(text, is_text=True)
        else:
            return
        
        st.session_state["search_results"] = vector_db.search(embedding, num_results)
    except Exception as e:
        st.error(f"Search error: {e}")

# Rerank results based on selected images
def handle_rerank(vector_db, selected_results, num_results):
    if not selected_results:
        return
    
    def process_single_image(path):
        img_bytes = load_and_resize_image(path)
        return get_cached_embedding(img_bytes)
    
    selected_results_list = list(selected_results)
    selected_embeddings = batch_process_images(selected_results_list, process_single_image)
    
    if selected_embeddings:
        # Average the embeddings of selected images
        selected_mean_embedding = np.mean(selected_embeddings, axis=0)
        results = vector_db.search(selected_mean_embedding, num_results)
        st.session_state["search_results"] = list(results) if isinstance(results, set) else results

# Cache base64 encoded images for faster display
@st.cache_data(ttl=3600)
def get_image_base64(image_path: str) -> str:
    img_bytes = load_and_resize_image(image_path)
    if img_bytes:
        return base64.b64encode(img_bytes).decode()
    return None

# Display images in a nice grid layout
def display_results_in_grid(results, columns=3, selectable=False, key_prefix=""):
    results_list = list(results) if isinstance(results, set) else results
    
    for i in range(0, len(results_list), columns):
        cols = st.columns(columns)
        batch_results = results_list[i:min(i + columns, len(results_list))]
        
        # Pre-load all images in current batch
        image_data = {
            result: get_image_base64(result) 
            for result in batch_results
        }
        
        for col, result in zip(cols[:len(batch_results)], batch_results):
            with col:
                if image_data[result]:
                    st.markdown(
                        f'<img src="data:image/jpeg;base64,{image_data[result]}" style="width:100%">',
                        unsafe_allow_html=True
                    )
                    if selectable:
                        checkbox_key = f"{key_prefix}_{result}"
                        is_selected = result in st.session_state.get("selected_images", set())
                        st.checkbox(
                            "Select",
                            key=checkbox_key,
                            value=is_selected,
                            on_change=update_selection,
                            args=(result, not is_selected)
                        )

# Load and cache the vector database
@st.cache_resource(ttl=3600)
def load_vector_database(index_path):
    try:
        vector_db = VectorStore()
        vector_db.load(index_path)
        return vector_db
    except Exception as e:
        st.error(f"Failed to load vector database: {e}")
        return None

# Initialize the app
vector_db = load_vector_database('fashion_dataset/index')

st.title("Visual Search for Online Shopping")

# Setup session state
if "search_results" not in st.session_state:
    st.session_state["search_results"] = []
if "selected_images" not in st.session_state:
    st.session_state["selected_images"] = set()
if "page" not in st.session_state:
    st.session_state["page"] = "Search"

# Navigation sidebar
st.session_state["page"] = st.sidebar.radio("Navigation", ["Search", "View Selections"])

# Search page UI
if st.session_state["page"] == "Search":
    st.subheader("Search")
    left, right = st.columns(2)
    
    uploaded_file = left.file_uploader(
        "**Upload an image:**",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )
    
    text_query = right.text_input(
        "**Or enter a textual description:**",
        key="text_query"
    )
    
    num_results = st.number_input(
        "Number of results to display:",
        min_value=1,
        max_value=100,
        value=5,
        key="num_results"
    )

    # Search and rerank buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Search", key="search_button"):
            if uploaded_file:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, use_container_width=True)
                handle_search(vector_db, image=image, num_results=num_results)
            elif text_query:
                handle_search(vector_db, text=text_query, num_results=num_results)
            else:
                st.warning("Please provide an image or a text query.")

    with col2:
        if st.session_state["selected_images"] and st.button("Rerank", key="rerank_button"):
            handle_rerank(
                vector_db,
                st.session_state["selected_images"],
                len(st.session_state["search_results"])
            )

    # Show search results
    if st.session_state["search_results"]:
        display_results_in_grid(
            st.session_state["search_results"],
            columns=5,
            selectable=True,
            key_prefix="search"
        )

# Selections page UI
if st.session_state["page"] == "View Selections":
    st.subheader("Your Selected Images")
    if st.session_state["selected_images"]:
        display_results_in_grid(
            list(st.session_state["selected_images"]),
            columns=3,
            selectable=True,
            key_prefix="selection"
        )
    else:
        st.info("No images selected yet.")