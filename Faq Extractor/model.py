import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Load FAQs jsonl 
@st.cache_data
def load_faqs(path="filtered_faqs.jsonl"):
    faqs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            faqs.append(json.loads(line))
    questions = [faq["question"] for faq in faqs]
    answers = [faq["answer"] for faq in faqs]
    embeddings = model.encode(questions, convert_to_tensor=True)
    return questions, answers, embeddings

questions, answers, question_embeddings = load_faqs()

# Search FAQs
def retrieve_answer(query, top_k=3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, question_embeddings, top_k=top_k)[0]

    results = []
    for hit in hits:
        idx = hit["corpus_id"]
        score = hit["score"]
        results.append({
            "question": questions[idx],
            "answer": answers[idx],
            "score": float(score)
        })
    return results

st.title("FAQ Retrieval (RAG with Sentence Transformers)")

query = st.text_input("Ask your question:")

if query:
    results = retrieve_answer(query, top_k=5)

    st.subheader("Top Answers")
    for r in results:
        st.markdown(f"**Q:** {r['question']}")
        st.markdown(f"**A:** {r['answer']}")
        st.caption(f"Score: {r['score']:.4f}")
        st.markdown("---")
