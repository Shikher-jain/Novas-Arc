from sentence_transformers import SentenceTransformer, util

st_model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "I like coffee",
    "I enjoy tea",
    "I love drinking coffee",
    "I want to learn programming",
    "I like eating sweets"
]

embeddings = st_model.encode(sentences, convert_to_tensor=True)

similarity_matrix = util.cos_sim(embeddings, embeddings)
print("Cosine Similarity Matrix:\n", similarity_matrix)

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
