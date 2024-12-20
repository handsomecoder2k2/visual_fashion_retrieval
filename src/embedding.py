from sentence_transformers import SentenceTransformer
from PIL import Image

model = SentenceTransformer('clip-ViT-B-32')

def get_image_embedding(image):
    return model.encode(image, convert_to_tensor=True)

def get_text_embedding(text):
    return model.encode(text, convert_to_tensor=True)


def get_embedding(query, is_text=False):
    if is_text:
        query_embedding = get_text_embedding(query)
    else:
        query_embedding = get_image_embedding(query)
    
    # Ensure the tensor is moved to CPU before converting to numpy
    query_embedding = query_embedding.cpu().numpy()

    return query_embedding
