import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading dataset...")

# Load dataset
df = pd.read_csv("dataset.csv")

# Limit dataset size to fit Render free tier
df = df.head(20000)

# Ensure text column exists
if "text" not in df.columns:
    raise ValueError("Dataset must contain a 'text' column")

texts = df["text"].astype(str).tolist()

print("Total sentences:", len(texts))

print("Loading Sentence-BERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    batch_size=64
)

print("Saving embeddings to file...")

np.save("embeddings.npy", embeddings)

print("Embeddings saved successfully!")
print("Embedding shape:", embeddings.shape)