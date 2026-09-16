from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from engine import get_results

app = FastAPI(title="Concept Shelf API")

# 1. Enable CORS so your browser is allowed to talk to localhost:8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (local files, localhost, etc.)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 2. The Search API endpoint that index.html calls
@app.get("/api/search")
def search(q: str):
    if not q.strip():
        return []
    # Calls your 2-stage engine
    return get_results(search_query=q, top_n=5)


# 3. Bonus: Automatically serve your index.html at http://localhost:8000/
@app.get("/")
def serve_ui():
    return FileResponse("index.html")