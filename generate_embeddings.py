import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading dataset...")

df = pd.read_csv("dataset.csv")

texts = df["text"].tolist()

print("Loading BERT model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")

embeddings = model.encode(texts, show_progress_bar=True)

print("Saving embeddings...")

np.save("embeddings.npy", embeddings)

print("Embeddings generated successfully!")
print("Total sentences:", len(texts))