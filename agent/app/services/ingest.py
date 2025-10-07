from openai import OpenAI
import chromadb
import uuid
from pathlib import Path


def simple_text_splitter(text: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """Split text into overlapping chunks without LangChain."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def ingest_document(
    doc_text: str,
    collection_name: str = "startup_mentor",
    db_path: str = "./chroma_db",
):
    """
    Ingest a document into a persistent Chroma collection.
    - Splits text into chunks
    - Embeds each chunk with OpenAI
    - Stores embeddings + chunks in Chroma
    """

    # Split into chunks
    chunks = simple_text_splitter(doc_text, chunk_size=500, chunk_overlap=50)

    if not chunks:
        print("No text to ingest.")
        return None

    # Initialize clients
    client = OpenAI()
    chroma = chromadb.PersistentClient(path=db_path)
    collection = chroma.get_or_create_collection(name=collection_name)

    # Create embeddings in batch
    response = client.embeddings.create(model="text-embedding-3-small", input=chunks)
    embeddings = [e.embedding for e in response.data]

    # Store in Chroma (unique IDs)
    ids = [str(uuid.uuid4()) for _ in chunks]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)

    print(f"Ingested {len(chunks)} chunks into collection '{collection_name}' at {db_path}")
    return collection


def ingest_from_dir(
    dir_path: str = "./docs",
    collection_name: str = "startup_mentor",
    db_path: str = "./chroma_db",
    file_types: list[str] = None,
):
    """
    Ingest all supported documents from a directory into ChromaDB.
    Supports `.txt` and `.mp3` (transcribes audio first).
    """
    if file_types is None:
        file_types = ["txt"]

    client = OpenAI()
    dir_path = Path(dir_path)

    if not dir_path.exists():
        print(f"Directory {dir_path} does not exist.")
        return None

    for file in dir_path.iterdir():
        if not file.is_file():
            continue

        ext = file.suffix.lower().lstrip(".")
        if ext not in file_types:
            continue

        print(f"Processing {file.name} ...")

        if ext == "txt":
            text = file.read_text(encoding="utf-8")
            ingest_document(text, collection_name=collection_name, db_path=db_path)

        elif ext == "mp3":
            with open(file, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=audio_file,
                )
                text = transcript.text
                ingest_document(text, collection_name=collection_name, db_path=db_path)
                print("mp3 Ingestion complete.")

    print("Ingestion complete.")
