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

print("loading data...")
books = pd.read_csv("clean_books.csv")
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
descriptions = books["combined_features"]

print("loading embeddings...")
embeddings = np.load("book_embeddings.npy", mmap_mode="r")
embeddings_gpu = torch.tensor(embeddings, device=device)

def get_results(search_query:str, top_n: int=5):

    scores = model.similarity(
        embeddings1=embeddings_gpu,
        embeddings2=torch.tensor(model.encode(search_query, prompt="query: ",  device=device).reshape(1, -1)).to(device)
    ).squeeze()

    top_scores, top_indicies = torch.topk(scores, top_n * 5)
    top_indicies = top_indicies.cpu().numpy()

    candidate_descriptions = descriptions.iloc[top_indicies].tolist()
    raw_cross_scores = cross_encoder.predict([(search_query, desc) for desc in candidate_descriptions])

    sorted_cross_scores = np.argsort(raw_cross_scores)[::-1]
    top_cross_indicies = sorted_cross_scores[:top_n]
    top_scores = raw_cross_scores[top_cross_indicies].tolist()
    final_indicies = top_indicies[top_cross_indicies]

    recommendations = []
    for i in range(top_n):
        idx = final_indicies[i]
        book_data = {
            "rank": i+1,
            "name": books["Name"].iloc[idx],
            "score": round(float(sigmoid(top_scores[i]+3) * 100), 4),
            "Id": books["Id"].iloc[idx]
        }
        recommendations.append(book_data)

    return recommendations

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


# while True:
#     print('-' * 50)
#     results = get_results(input("What type of book are you looking for? "), 10)
#     for book in results:
#         print('-' * 50)
#         print(f"{book['rank']}: {book['name']} | score: {book['score']}%")