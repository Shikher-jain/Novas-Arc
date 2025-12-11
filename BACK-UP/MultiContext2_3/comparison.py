from sentence_transformers import SentenceTransformer, util
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

model = SentenceTransformer('all-MiniLM-L6-v2')

model_text = """
Internet gateways are highly available and automatically scale to meet your demand. There are no bandwidth constraints or limitations for Internet gateways.
"""
site_text = """
No, An Internet gateway is horizontally-scaled, redundant, and highly available. It imposes no bandwidth constraints.
"""

embeddings = model.encode([model_text, site_text])
similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

print(f"Semantic Similarity: {similarity:.2f}")

if similarity > 0.80:
    print("Same meaning.")
elif similarity > 0.6:
    print("Related.")
else:
    print("Different meaning.")

