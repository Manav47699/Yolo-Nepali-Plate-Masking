import json
import os
import shutil

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

JSON_PATH = "master_database.json"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "nepali_foods"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ---------------------------------------------------------
# BUILD VECTOR DATABASE
# ---------------------------------------------------------

def build_vector_store():

    # 1. Check JSON file
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(
            f"Could not find '{JSON_PATH}'."
        )

    # 2. Load master database
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    foods = data.get("foods")

    if not isinstance(foods, list):
        raise ValueError(
            "master_database.json must contain a top-level "
            "'foods' array."
        )

    if len(foods) == 0:
        raise ValueError(
            "The 'foods' array is empty."
        )

    print(f"Found {len(foods)} food items.")


    # -----------------------------------------------------
    # 3. Create documents
    # -----------------------------------------------------

    documents = []

    for food in foods:

        # Required fields
        food_id = food["id"]
        name = food["name"]

        other_names = food.get(
            "other_names", []
        )

        # -------------------------------------------------
        # TEXT THAT GETS EMBEDDED
        #
        # Keep this focused on identifying the food.
        # -------------------------------------------------

        aliases = ", ".join(other_names)

        page_content = (
            f"Food name: {name}\n"
            f"Food ID: {food_id}\n"
            f"Other names and aliases: {aliases}"
        )


        # -------------------------------------------------
        # METADATA
        #
        # This information is stored with the vector but
        # does NOT need to influence semantic matching.
        # -------------------------------------------------

        nutrition = food.get(
            "nutrition_per_gram", {}
        )

        health = food.get(
            "health_restrictions", {}
        )

        social = food.get(
            "social_restrictions", {}
        )

        metadata = {
            "id": food_id,

            "name": name,

            "other_names": json.dumps(
                other_names,
                ensure_ascii=False
            ),

            "veg_or_nonveg": food.get(
                "veg_or_nonveg",
                ""
            ),

            "fitness_direction": food.get(
                "fitness_direction",
                ""
            ),

            "nutrition_per_gram": json.dumps(
                nutrition,
                ensure_ascii=False
            ),

            "health_restrictions": json.dumps(
                health,
                ensure_ascii=False
            ),

            "social_restrictions": json.dumps(
                social,
                ensure_ascii=False
            ),
        }


        # -------------------------------------------------
        # Create LangChain document
        # -------------------------------------------------

        document = Document(
            page_content=page_content,
            metadata=metadata,
            id=food_id
        )

        documents.append(document)


    # -----------------------------------------------------
    # 4. Validate duplicate IDs
    # -----------------------------------------------------

    ids = [doc.id for doc in documents]

    if len(ids) != len(set(ids)):
        duplicates = {
            x for x in ids
            if ids.count(x) > 1
        }

        raise ValueError(
            f"Duplicate food IDs found: {duplicates}"
        )


    # -----------------------------------------------------
    # 5. Delete old Chroma database
    #
    # This ensures the vector database always exactly
    # matches master_database.json.
    # -----------------------------------------------------

    if os.path.exists(CHROMA_PATH):

        print(
            f"Removing old Chroma database: "
            f"{CHROMA_PATH}"
        )

        shutil.rmtree(CHROMA_PATH)


    # -----------------------------------------------------
    # 6. Load multilingual embedding model
    # -----------------------------------------------------

    print(
        f"Loading embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


    # -----------------------------------------------------
    # 7. Create Chroma vector database
    # -----------------------------------------------------

    print("Creating embeddings...")

    vectorstore = Chroma.from_documents(
        documents=documents,

        embedding=embedding_model,

        persist_directory=CHROMA_PATH,

        collection_name=COLLECTION_NAME
    )


    # -----------------------------------------------------
    # 8. Done
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("VECTOR DATABASE CREATED SUCCESSFULLY")
    print("=" * 60)

    print(f"Food items:     {len(documents)}")
    print(f"Collection:     {COLLECTION_NAME}")
    print(f"Database path:  {os.path.abspath(CHROMA_PATH)}")
    print(f"Embedding:      {EMBEDDING_MODEL}")
    print("=" * 60)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":
    build_vector_store()