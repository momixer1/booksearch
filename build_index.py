import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

print("loading...")
model = SentenceTransformer('intfloat/multilingual-e5-small', local_files_only=False, device=device)
books = pd.read_csv("clean_books.csv")

print("preparing...")
max_words_desc = 128
books['Description'] = books["Description"].fillna('').apply(
    lambda desc: ' '.join(desc.split(' ')[:max_words_desc]) + ('...' if len(desc.split(' ')) > max_words_desc else desc))

books["combined_features"] = (
    "Title: " +
    books["Name"] +
    "| Author: " +
    books["Authors"] +
    "| Year: " +
    books["PublishYear"].astype(str) +
    "| Description: " +
    books["Description"]
)

print("encoding:")
embeddings = model.encode(books["Description"].tolist(), show_progress_bar=True, batch_size=64, device=device)
print(f"embedding shape: {embeddings.shape}")

print("saving...")
np.save("book_embeddings.npy", embeddings)
print("Embeddings calculated and saved successfully.")