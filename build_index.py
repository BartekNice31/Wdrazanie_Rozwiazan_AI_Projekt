"""
Buduje indeks ChromaDB z pliku CSV z ogłoszeniami.
Uruchom raz przed startem API:  python build_index.py
"""

import re
import csv
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from openai import AuthenticationError
import pandas as pd
 

def clean_int(x):
    if x is None:
        return None
    try:
        return int(float(str(x).replace(" ", "")))
    except:
        return None

def clean_float(x):
    if x is None:
        return None
    try:
        return float(str(x).replace(" ", "").replace("zł", "").replace(",", "."))
    except:
        return None

def clean_floor(x):
    if x is None:
        return None
    x = str(x).lower()

    if "parter" in x:
        return 0
    try:
        return int(re.findall(r"\d+", x)[0])
    except:
        return None

def clean_yes_no(x):
    if x is None:
        return None
    x = str(x).lower()
    if x in ["tak", "yes", "true", "1"]:
        return True
    if x in ["nie", "no", "false", "0"]:
        return False
    return None

BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))
# load_dotenv(os.path.join(BASE_DIR, "env.txt"))
CSV_PATH = os.path.join(BASE_DIR, "ogloszenia_warszawa_detailed.csv")

COLLECTION_NAME = "ogloszenia"
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_index")

CLEAN_CSV_PATH = os.path.join(BASE_DIR, "ogloszenia_warszawa_detailed_v3.csv")

print(CHROMA_PATH)
print(os.path.exists(CHROMA_PATH))

api_key = os.getenv("OPENAI_API_KEY")
print(api_key[:10])

TEXT_FIELDS=['locality',
    'street',
    'owner_type',#?? to są kategorie / atrybuty, nie tekst do „rozumienia”
    'city_district',
    'full_address',
    'building_type',
    'description_text',
    'kitchen_type',
    'ownership_type',#?? to są kategorie / atrybuty, nie tekst do „rozumienia”
    'equipment']

META_FIELDS=['locality',#może mieszać w logice bo się powtarza w TEXT_FIELDS
    'street', #może mieszać w logice bo się powtarza w TEXT_FIELDS
    'rooms',
    'area',
    'price_total_zl',
    'price_sqm_zl',
    'owner_type',
    'url',
    'image_url',
    'city_district', #może mieszać w logice bo się powtarza w TEXT_FIELDS
    'floor',
    'year_built',
    'has_basement',
    'has_parking',
    'latitude',
    'longitude']

def load_documents_from_csv(path: str) -> list[Document]:
    docs = []

    with open(path, encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):

            # TEXT (embedding)
            parts = [row.get(f, "") or "" for f in TEXT_FIELDS]
            content = " | ".join(p for p in parts if p) # Use raw values for text content

            # META (NORMALIZED) - Use clean_row for all metadata
            metadata = clean_row(row)
            metadata["id"] = str(i) # Add ID to metadata

            docs.append(Document(
                page_content=content,
                metadata=metadata
            ))

    return docs
def save_clean_csv(input_path: str, output_path: str):
    print(">>> SAVE CLEAN CSV START")
    try:
        cleaned_rows = []

        with open(input_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cleaned_rows.append(clean_row(row))

        df = pd.DataFrame(cleaned_rows)

        print("DEBUG DF SHAPE:", df.shape)

        df.to_csv(output_path, index=False, encoding="utf-8")

        print("Zapisano clean CSV:", output_path)

    except Exception as e:
        print("ERROR CSV SAVE:", e)
        raise 
    
def clean_row(row: dict) -> dict:
    return {
        "locality": row.get("locality", ""),
        "street": row.get("street", ""),
        "city_district": row.get("city_district", ""),
        "owner_type": row.get("owner_type", ""),
        "url": row.get("url", ""),
        "image_url": row.get("image_url", ""),
        "description_text": row.get("description_text", ""),

        "rooms": clean_int(row.get("rooms")),
        "area": clean_int(row.get("area")),
        "price_total_zl": clean_float(row.get("price_total_zl")),
        "price_sqm_zl": clean_float(row.get("price_sqm_zl")),
        "floor": clean_floor(row.get("floor")),
        "year_built": clean_float(row.get("year_built")),
        "latitude": clean_float(row.get("latitude")),
        "longitude": clean_float(row.get("longitude")),

        "has_parking": clean_yes_no(row.get("has_parking")),
        "has_basement": clean_yes_no(row.get("has_basement")),
    }
def build(force: bool = False):
    if os.path.exists(CHROMA_PATH) and not force:
        print(f"Indeks już istnieje: {CHROMA_PATH}  (użyj --force żeby przebudować)")
        return

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAPI_KEY")
    if not api_key:
        raise RuntimeError("Brak OPENAI_API_KEY w środowisku lub pliku .env") 

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAPI_KEY") 

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key
    ) 

    print("Czyszczenie CSV...")
    save_clean_csv(CSV_PATH, CLEAN_CSV_PATH)
    print("CSV exists:", os.path.exists(CLEAN_CSV_PATH))
    print("ABS:", os.path.abspath(CLEAN_CSV_PATH))

    print("Ładowanie CLEAN CSV...")
    docs = load_documents_from_csv(CLEAN_CSV_PATH)

    print("Docs:", len(docs))
    print("DEBUG metadata example:", docs[0].metadata)

    vectorstore = None

    for start in range(0, len(docs), 500):
        batch = docs[start:start+500]

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                batch,
                embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=CHROMA_PATH,
            )
        else:
            vectorstore.add_documents(batch) 
    print("DONE")
if __name__ == "__main__":
    import sys
    build(force="--force" in sys.argv)