import faiss
import numpy as np
import torch
import json

class VectorStore:
    def __init__(self, dimension=512, index_path='fashion_dataset/index'):
        self.index = faiss.IndexFlatL2(dimension)
        self.image_paths = []
        self.index_path = index_path

    def load(self, index_path):
        self.index = faiss.read_index(index_path + '/faiss.index')
        with open(index_path + '/metadata.json', 'r') as file:
            self.image_paths = json.load(file)

    def add(self, embeddings, paths):
        # Convert embeddings to numpy arrays
        embeddings = [embedding.cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding for embedding in embeddings]
        self.index.add(np.array(embeddings))
        self.image_paths.extend(paths)

    def search(self, query_embedding, k=5):
        if k <= 0:
            k = 1
        distances, indices = self.index.search(np.array([query_embedding]), k)
        print(len(self.image_paths))
        return [self.image_paths[i] for i in indices[0]]
