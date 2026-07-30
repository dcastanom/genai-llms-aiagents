from sentence_transformers import SentenceTransformer  # pyright: ignore[reportMissingImports]
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# all-MiniLM-L6-v2 is a small, fast, and efficient model for generating sentence embeddings.
print("Loading the sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully.")

sentences = [
    "A dog is chasing a bright red ball on a sunny green field.",
    "The canine pursues a crimson sphere across the lawn.",
    "I am planning to bake an apple pie for the picnic next weekend.",
    "A new laptop was purchaased from the tech store today.",
]

# generate embeddings for the sentences
print("Generating embeddings for the sentences...")
embeddings = model.encode(sentences)
print("Embeddings generated successfully.")

# check the shape and size of the embeddings
print("Shape of embeddings:", embeddings.shape)
print("Shape of embeddings:", embeddings.shape[0], "x", embeddings.shape[1])
print("Size of embeddings:", embeddings.size)
print("Sample of the first embedding vector:", embeddings[0][:10])  # print first 10 dimensions of the first embedding

# calculate cosine similarity matrix for all pairs of sentences
print("Calculating cosine similarity matrix...")
similarity_matrix = cosine_similarity(embeddings)  # inner product gives cosine similarity for normalized vectors
print("Cosine similarity matrix calculated successfully.")
print("Cosine similarity matrix:\n", similarity_matrix)

print("\nPairwise cosine similarity between sentences: \n score = 1 identical, score = 0 orthogonal, score = -1 opposite meaning")

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        print(f"Similarity between \"{sentences[i]}\" and \"{sentences[j]}\": {similarity_matrix[i][j]:.4f}")


