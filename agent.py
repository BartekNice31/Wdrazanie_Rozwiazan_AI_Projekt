import os
import asyncio
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import create_react_agent

from build_index import BASE_DIR, CSV_PATH, CHROMA_PATH, COLLECTION_NAME, load_documents_from_csv
from search_engine import build_retrievers, search_documents

load_dotenv(os.path.join(BASE_DIR, ".env"))

# GLOBAL_MODEL_EMBEDDINGS='text-embedding-3-large'
GLOBAL_MODEL_EMBEDDINGS='text-embedding-3-small'

def _format_docs(docs: list) -> str:
    parts = []
    for i, (doc, score) in enumerate(docs, 1):
        score_str = f" (score: {score:.2f})" if score is not None else ""
        parts.append(f"[{i}]{score_str}\n{doc.page_content}\nMetadata: {doc.metadata}")
    return "\n\n".join(parts)

def build_agent(api_key: str, vectorstore, docs: list, retrievers: dict
                # ,top_k: int = 3
                ):
    @tool
    async def szukaj_mieszkania(query: str, top_k: int = 3) -> str:
        """Wyszukuje ogłoszenia mieszkaniowe w Warszawie na podstawie opisu.

        Args:
            query: Opis szukanego mieszkania, np. "Mieszkanie z 3 pokojami w dzielnicy Wola"
            top_k: Liczba wyników do zwrócenia (domyślnie 3)
        """
        result_docs = await search_documents(query, "hybrid", top_k, None, vectorstore, docs, retrievers)
        if not result_docs:
            return "Nie znaleziono ogłoszeń pasujących do zapytania."
        return _format_docs(result_docs)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    return create_react_agent(
        model=llm,
        tools=[szukaj_mieszkania],
        prompt=(
            "Jesteś pomocnym asystentem do wyszukiwania mieszkań w Warszawie. "
            "Używaj narzędzia szukaj_mieszkania, żeby znaleźć oferty pasujące do potrzeb użytkownika. "
            "Odpowiadaj po polsku i podsumuj wyniki w czytelny sposób."
        ),
    )
async def run_agent_query(agent, message: str) -> str:
    """Wywołuje agenta z podaną wiadomością i zwraca tekstową odpowiedź."""
    result = await agent.ainvoke({"messages": [{"role": "user", "content": message}]})
    return result["messages"][-1].content

if __name__ == "__main__":
    async def _main():
        api_key = os.getenv("OPENAI_API_KEY")
        # embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
        #text-embedding-3-large
        embeddings = OpenAIEmbeddings(model=GLOBAL_MODEL_EMBEDDINGS, api_key=api_key)
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PATH,
        )
        docs = load_documents_from_csv(CSV_PATH)
        retrievers = build_retrievers(vectorstore, docs)
        agent = build_agent(api_key, vectorstore, docs, retrievers)

        user_message = "Znajdź mi mieszkanie 3-pokojowe na ulicy Stanisława Staszica do 700 000 PLN"
        print(f"Pytanie: {user_message}\n")
        print(await run_agent_query(agent, user_message))

    asyncio.run(_main())