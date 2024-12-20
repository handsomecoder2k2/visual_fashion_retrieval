import os
import numpy as np
import faiss
from PIL import Image
from tqdm import tqdm
from src.embedding import get_image_embedding
from src.index import VectorStore
from src.config import DATASET_PATH, INDEX_PATH
import json

# Step 1: Initialize the Vector Store
dimension = 512  # Adjust to match the embedding dimension of your model
vector_store = VectorStore(dimension=dimension)

# Step 2: Load and Process Dataset
def load_images(dataset_path):
    image_paths = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))

    return image_paths

image_paths = load_images(DATASET_PATH)

# Step 3: Generate Embeddings and Add to Vector Store
print(f"Indexing {len(image_paths)} images from the dataset...")
embeddings = []
paths = []

for path in tqdm(image_paths):
    try:
        # Load image and extract embedding
        image = Image.open(path).convert('RGB')
        embedding = get_image_embedding(image).cpu().numpy()  # Ensure embedding is on CPU
        embeddings.append(embedding)
        paths.append(path)
    except Exception as e:
        print(f"Error processing {path}: {e}")

# Add embeddings and paths to the vector store
# embeddings = np.vstack(embeddings)
if len(embeddings) == 0:
    print("No valid embeddings found. Please check your dataset or embedding function.")
    exit()
vector_store.add(embeddings, paths)
with open('fashion_dataset/index/metadata.json', 'w') as file:
    json.dump(paths, file)

# Step 4: Save the FAISS Index
os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
faiss.write_index(vector_store.index, INDEX_PATH)
print(f"Index saved to {INDEX_PATH}")
