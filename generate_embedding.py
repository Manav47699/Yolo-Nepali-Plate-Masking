# this code converts .json into embedding file
import json
import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def build_vector_store():
    # 1. Load the food dataset
    json_path = "sample_master.json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find {json_path} in current directory.")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    foods = data.get("foods", [])

    # 2. Build rich documents for multilingual semantic matching
    documents = []
    for food in foods:
        food_id = food["id"]
        name = food["name"]
        other_names = food.get("other_names", [])

        # Format a high-density description string incorporating English, Romanized Nepali, and Devanagari names
        aliases_str = ", ".join(other_names)
        content = (
            f"Food ID: {food_id}\n"
            f"Primary Name: {name}\n"
            f"Alternative and Local Names: {aliases_str}"
        )

        # Store complete metadata so it can be queried directly later
        metadata = {
            "id": food_id,
            "name": name,
            "other_names": ",".join(other_names),
            "nutrition_per_gram": json.dumps(food.get("nutrition_per_gram", {})),
            "health_restrictions": json.dumps(food.get("health_restrictions", {})),
            "social_restrictions": json.dumps(food.get("social_restrictions", {})),
        }

        documents.append(Document(page_content=content, metadata=metadata, id=food_id))

    # 3. Initialize a high-quality multilingual embedding model (ideal for English + Nepali transliterations)
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 4. Generate and persist vector embeddings locally into ChromaDB directory
    persist_directory = "./chroma_db"
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )

    print(
        f"Successfully converted {len(documents)} food items into vector embeddings!"
    )
    print(f"Saved local ChromaDB database to: {os.path.abspath(persist_directory)}")


if __name__ == "__main__":
    build_vector_store()