from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
load_dotenv()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

Documents =[
    "Kenil Bhadani Is a retailer and a business owner",
    "Ram is a software engineer and a tech enthusiast",
    "Sita is a doctor and a healthcare professional",
    "Ravi is a teacher and an education advocate",
    "Anita is a lawyer and a legal consultant",
]

query = "Tell me about Kenil Bhadani"

document_embeddings = embeddings.embed_documents(Documents)
query_embedding = embeddings.embed_query(query)

similarities = cosine_similarity([query_embedding], document_embeddings)

print(list(enumerate(similarities[0])))

