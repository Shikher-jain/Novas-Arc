model_response = """
When you launch an instance into a subnet, the instance will be assigned an IP address from that subnet. If you want to assign additional private IP addresses to your network interface, see Adding secondary IPv4 addresses (Private IPv4 Addresses) to or removing secondary IPv4 addresses from your network interface.
"""
site_response = """
When you launch an Amazon EC2 instance within a subnet that is not IPv6-only, you may optionally specify the primary private IPv4 address for the instance. If you do not specify the primary private IPv4 address, AWS automatically addresses it from the IPv4 address range you assign to that subnet. You can assign secondary private IPv4 addresses when you launch an instance, when you create an Elastic Network Interface, or any time after the instance has been launched or the interface has been created. In case you launch an Amazon EC2 instance within an IPv6-only subnet, AWS automatically addresses it from the Amazon-provided IPv6 GUA CIDR of that subnet. The instance’s IPv6 GUA will remain private unless you make them reachable to/from the internet with the right security group, NACL, and route table configuration.
"""

# import spacy
import subprocess
import sys
import warnings

# Silence transformers FutureWarning about `clean_up_tokenization_spaces`
# (emitted indirectly via sentence-transformers/transformers stack)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*clean_up_tokenization_spaces.*",
)

# def load_spacy_model(model_name="en_core_web_sm"):
#     try:
#         # Try loading the model
#         nlp = spacy.load(model_name)
#         print(f"Loaded spaCy model '{model_name}'")
#     except OSError:
#         # Model not found, download it
#         print(f"Model '{model_name}' not found. Downloading now...")
#         subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)
#         nlp = spacy.load(model_name)
#         print(f"Downloaded and loaded spaCy model '{model_name}'")
#     return nlp

# Usage
# nlp = load_spacy_model()

# # Test semantic similarity
# doc1 = nlp(model_response)
# doc2 = nlp(site_response)
# print("Similarity from spaCy               : ", sim_1:=doc1.similarity(doc2))


from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
emb1 = model.encode(model_response, convert_to_tensor=True)
emb2 = model.encode(site_response, convert_to_tensor=True)
print("Similarity from SentenceTransformers: ",sim_2:=util.cos_sim(emb1, emb2).item())

# print(f"{((sim_1 + sim_2)/2)*100}")
# print(f"Approx                             :  {(sim_1 + sim_2)*50}")