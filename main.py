import asyncio
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from build_index import load_documents_from_csv
from search_engine import SearchRequest, SearchResult, SearchResponse, search_documents, build_retrievers
from build_index import CHROMA_PATH, COLLECTION_NAME, CSV_PATH, BASE_DIR
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from agent import build_agent, run_agent_query,GLOBAL_MODEL_EMBEDDINGS

load_dotenv(os.path.join(BASE_DIR, ".env"))

state: dict = {}
retrievers: dict = {}
    


class AgentRequest(BaseModel):
    message: str

class AgentResponse(BaseModel):
    answer: str

def get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment or .env")
    return api_key



@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(CHROMA_PATH):
        raise RuntimeError(
            f"Indeks ChromaDB nie istnieje: {CHROMA_PATH}\n"
            "Uruchom najpierw: python build_index.py"
        )

    embeddings = OpenAIEmbeddings(
        model=GLOBAL_MODEL_EMBEDDINGS,
        api_key=get_openai_api_key(),
    )
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )
    docs = load_documents_from_csv(CSV_PATH)

    state["vectorstore"] = vectorstore
    state["embeddings"] = embeddings
    state["docs"] = docs
    state["llm"] = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=get_openai_api_key())
    retrievers.update(build_retrievers(vectorstore, state["docs"]))
    state["agent"] = build_agent(get_openai_api_key(), vectorstore, state["docs"], retrievers)
    yield
    
app = FastAPI(title="Search API", lifespan=lifespan)

@app.get("/")
def root():
    return {"status":"ok"}
    
@app.get("/health")
async def health():
    return {"status OK at time:": f"{str(datetime.datetime.now())}"}

@app.post("/rag/search", response_model=SearchResponse)
async def search_docs(req: SearchRequest):
    if req.top_k < 1 or req.top_k > 15:
        raise HTTPException(status_code=422, detail="top_k (Liczba wyników) musi być wartością pomiędzy 1 a 15")
    try:
        docs = await search_documents(
            req.query, req.mode, req.top_k, req.filters,
            state["vectorstore"], state["docs"], retrievers,
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return SearchResponse(
        query=req.query,
        mode=req.mode,
        results=[SearchResult(content=doc.page_content, metadata=doc.metadata, score=score) for doc, score in docs],
    )
    docs = await search_documents(...)
    for doc, score in docs[:5]:
        print(doc.metadata)
    
@app.post("/rag/answer", response_model=AgentResponse)
async def answer_agent(req: AgentRequest):
    try:
        answer = await run_agent_query(state["agent"], req.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return AgentResponse(answer=answer)