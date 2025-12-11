from sentence_transformers import SentenceTransformer, util

# 1️⃣ Load the pre-trained model
st_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2️⃣ Define sentences
sentences = [
    "I like coffee",
    "I enjoy tea",
    "I love drinking coffee",
    "I want to learn programming",
    "I like eating sweets"
]

# 3️⃣ Convert sentences to embeddings (as tensors)
embeddings = st_model.encode(sentences, convert_to_tensor=True)

# 4️⃣ Compute cosine similarity matrix using util.cos_sim
similarity_matrix = util.cos_sim(embeddings, embeddings)
print("Cosine Similarity Matrix:\n", similarity_matrix)

# 5️⃣ Find top N similar sentences for a query
query_index = 0  # "I like coffee"
top_n = 2        # number of similar sentences to show

# Get similarities for the query sentence
query_similarities = similarity_matrix[query_index]

# Exclude the query itself
query_similarities[query_index] = -1  # set self-similarity to -1

# Get top N indices
top_indices = query_similarities.topk(top_n).indices

print(f"\nTop {top_n} sentences most similar to '{sentences[query_index]}':")
for i, idx in enumerate(top_indices):
    print(f"{i+1}. '{sentences[idx]}' (similarity: {query_similarities[idx]:.2f})")
