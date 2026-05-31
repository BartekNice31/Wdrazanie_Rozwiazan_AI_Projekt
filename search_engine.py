from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from pydantic import BaseModel, Field
import os
import asyncio
from typing import Literal

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings 

class MetadataFilter(BaseModel):
    cena_minimalna: float | None = Field(None, description="Minimalna cena (PLN)", examples=[400000])
    cena_maksymalna: float | None = Field(None, description="Maksymalna cena (PLN)", examples=[800000])
    liczba_pokoi_min: int | None = Field(None, description="Minimalna liczba pokoi", examples=[2])
    liczba_pokoi_max: int | None = Field(None, description="Maksymalna liczba pokoi", examples=[4])
    powierzchnia_mieszkania_min: float | None = Field(None, description="Minimalna powierzchnia mieszkania (m²)", examples=[35])
    powierzchnia_mieszkania_max: float | None = Field(None, description="Maksymalna powierzchnia mieszkania (m²)", examples=[120])
    cena_metr_kwadratowy_min: float | None = Field(None,description="Cena minimalna za metr kwadratowy powierzchni (m²)", examples=[6000])
    cena_metr_kwadratowy_max: float | None = Field(None,description="Cena maksymalna za metr kwadratowy powierzchni (m²)", examples=[12000])
    pietro_min: int | None = Field(None, description="Minimalne piętro", examples=[0])
    pietro_max: int | None = Field(None, description="Maksymalne piętro", examples=[31])
    rok_budownictwa_min: float | None = Field(None, description="Minimalny rok budownictwa", examples=[1900])
    rok_budownictwa_max: float | None = Field(None, description="Maksymalny rok budownictwa", examples=[2026])
    # dzielnica_miasta:str | None = Field(None,description="Dzielnica lub osiedle w Warszawie",examples=['Warszawa Wola'])
    
class SearchRequest(BaseModel):
    query: str = Field(..., examples=["mieszkanie w Warszawa Wola"])
    mode: Literal["semantic", "keyword", "hybrid"] = Field("hybrid", examples=["hybrid"])
    top_k: int = Field(default=3,
                        ge=1,
                        le=100,
                        description="Liczba wyników do zwrócenia",
                        examples=[5])
    filters: MetadataFilter | None = None

class SearchResult(BaseModel):
    content: str
    metadata: dict
    score: float | None = None

class SearchResponse(BaseModel):
    query: str
    mode: str
    results: list[SearchResult] 

def _to_float(val) -> float | None:
    try:
        return float(val) if val not in (None, "") else None
    except (ValueError, TypeError):
        return None
    
# def check_range(v, min_val, max_val):
#     if v is None:
#         return False

#     if min_val is not None and v < min_val:
#         return False

#     if max_val is not None and v > max_val:
#         return False

#     return True

# def check_range(v, min_val, max_val):
#     if v is None:
#         return False

#     if min_val is not None and v < min_val:
#         return False

#     if max_val is not None and v > max_val:
#         return False

#     return True

def passes_filter(meta: dict, f: MetadataFilter) -> bool:
    checks: list[bool] = []
    for field, min_val, max_val in [
        ("price_total_zl", f.cena_minimalna, f.cena_maksymalna),
        ("rooms", f.liczba_pokoi_min, f.liczba_pokoi_max),
        ("area", f.powierzchnia_mieszkania_min, f.powierzchnia_mieszkania_max),
        # ("price_sqm_zl",f.cena_metr_kwadratowy_min,f.cena_metr_kwadratowy_max),
        # ("floor",f.pietro_min,f.pietro_max),
        # ("year_built",f.rok_budownictwa_min,f.rok_budownictwa_max)
    ]:
        v = _to_float(meta.get(field))
        if min_val is not None:
            checks.append(v is not None and v >= min_val)
        if max_val is not None:
            checks.append(v is not None and v <= max_val) 
 
    return all(checks)

async def fetch_filtered(
    query: str,
    mode: str,
    fetch_k: int,
    f: MetadataFilter,
    top_k: int,
    vectorstore: Chroma,
    docs_all: list[Document],
) -> list[tuple[Document, float | None]]:
    if mode == "semantic":
        candidates: list[tuple[Document, float | None]] = await vectorstore.asimilarity_search_with_relevance_scores(query, k=fetch_k)
    elif mode == "keyword":
        bm25 = BM25Retriever.from_documents(docs_all, k=fetch_k)
        raw = await asyncio.to_thread(bm25.invoke, query)
        candidates = [(doc, None) for doc in raw]
    else:  # hybrid
        bm25 = BM25Retriever.from_documents(docs_all, k=fetch_k)
        sem_results, kw_docs = await asyncio.gather(
            vectorstore.asimilarity_search_with_relevance_scores(query, k=fetch_k),
            asyncio.to_thread(bm25.invoke, query),
        )
        seen_ids: set[str] = set()
        candidates = []
        for doc, score in sem_results:
            doc_id = doc.metadata.get("id", "")
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                candidates.append((doc, score))
        for doc in kw_docs:
            doc_id = doc.metadata.get("id", "")
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                candidates.append((doc, None))
    print("CANDIDATES:", len(candidates))

    for i, (doc, score) in enumerate(candidates[:5]):
        print(f"\n--- DOC {i} ---")
        print("SCORE:", score)
        print("META:", doc.metadata)
    filtered=[(d, s) for d, s in candidates if passes_filter(d.metadata, f)][:top_k]
    print("FILTERED:", len(filtered))
    if not filtered:
    # fallback: zwróć top_k bez filtrów
        return candidates[:top_k]
    return filtered

def build_retrievers(vectorstore: Chroma, docs: list[Document],top_k:int=4) -> dict:
    semantic = vectorstore.as_retriever(search_kwargs={"k": top_k})
    keyword = BM25Retriever.from_documents(docs, k=top_k)
    hybrid = EnsembleRetriever(
        retrievers=[semantic, keyword],
        weights=[0.6, 0.4],
    )
    return {"semantic": semantic, "keyword": keyword, "hybrid": hybrid}

FILTER_FETCH_MULTIPLIER = 15 

async def search_documents(
    query: str,
    mode: str,
    top_k: int,
    filters: "MetadataFilter | None",
    vectorstore: Chroma,
    docs_all: list[Document],
    retrievers: dict,
) -> list[tuple[Document, float | None]]:
    if filters is not None:
        fetch_k = top_k * FILTER_FETCH_MULTIPLIER
        return await fetch_filtered(query, mode, fetch_k, filters, top_k, vectorstore, docs_all)
    if mode == "semantic":
        results = await vectorstore.asimilarity_search_with_relevance_scores(query, k=top_k)
        return results[:top_k]
    retriever = retrievers.get(mode)
    if retriever is None:
        raise ValueError(f"Unknown retrieval mode: {mode}")
    raw = await retriever.ainvoke(query)
    return [(doc, None) for doc in raw[:top_k]]
