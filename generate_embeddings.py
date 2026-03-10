import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading dataset...")

df = pd.read_csv("dataset.csv")

texts = df["text"].astype(str).tolist()

print("Loading model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    batch_size=64
)

np.save("embeddings.npy", embeddings)

print("Embeddings saved as embeddings.npy")