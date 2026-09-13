from sentence_transformers import SentenceTransformer, CrossEncoder
import pandas as pd
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer('intfloat/multilingual-e5-small', local_files_only=True, device=device)
model.prompts = {"default": "passage: ",
                 "query": "query: "}
model.default_prompt_name = "query"
cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', local_files_only=True, device=device)

books = pd.read_csv("clean_books.csv")
num_books = len(books)
embedding_dim = 384 #standard
embeddings = np.load("book_embeddings.npy", mmap_mode="r")
embeddings_gpu = torch.from_numpy(embeddings).to(device)



def get_results(search_query:str, top_n: int=5):

    scores = model.similarity(
        embeddings1=embeddings_gpu,
        embeddings2=torch.tensor(model.encode(search_query, device=device).reshape(1, -1)).to(device)
    ).squeeze()

    top_scores, top_indicies = torch.topk(scores, top_n * 4)
    top_indicies = top_indicies.cpu().numpy()
    print(top_indicies)
    return books["Name"].iloc[top_indicies].tolist()

print("Ready")
while True:
    print(get_results(input(), 1))