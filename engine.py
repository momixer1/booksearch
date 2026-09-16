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

    query_emb = model.encode(search_query, prompt="query: ", convert_to_tensor=True, device=device)

    scores = model.similarity(embeddings_gpu, query_emb).squeeze()
    
    top_scores, top_indicies = torch.topk(scores, top_n * 5)
    top_indicies = top_indicies.cpu().numpy()

    candidate_descriptions = descriptions.iloc[top_indicies].tolist()
    raw_cross_scores = cross_encoder.predict([(search_query, desc) for desc in candidate_descriptions])

    sorted_cross_scores = np.argsort(raw_cross_scores)[::-1]

    percent_5s = books["RatingDist5"][top_indicies[sorted_cross_scores]]/books["RatingDistTotal"][top_indicies[sorted_cross_scores]]
    avg_ratings = np.asarray(books["Rating"][top_indicies[sorted_cross_scores]])

    relevance_prob = sigmoid(raw_cross_scores)
    combined = relevance_prob * ( 0.7 + 0.3 * sigmoid(avg_ratings) + 0.0 * percent_5s)[::-1]
    combined = np.asarray(combined)

    ranked_order = np.argsort(combined)[::-1]
    top_cross_indicies = np.asarray(ranked_order[:top_n])
    final_indicies = np.asarray(top_indicies)[top_cross_indicies]


    recommendations = []
    for i in range(top_n):
        cand_pos = top_cross_indicies[i]
        idx = final_indicies[i]
        match_percentage = round(float(sigmoid(raw_cross_scores+5)[cand_pos] * 100), 1)
        stars = round(float(avg_ratings[cand_pos]), 2)

        book_data = {
            "rank": int(i+1),
            "title": str(books["Name"].iloc[idx]),
            "author": str(books["Authors"].iloc[idx]),
            "year": str(books["PublishYear"].iloc[idx]),
            "score": f"{match_percentage}%",
            "lang": str(books["Language"].iloc[idx]),
            "desc": str(books["Description"].iloc[idx]),
            "rating": f"{stars} / 5.0",
            "Id": int(books["Id"].iloc[idx])
        }
        recommendations.append(book_data)

    return recommendations

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1 / (1 + np.exp(-x))


