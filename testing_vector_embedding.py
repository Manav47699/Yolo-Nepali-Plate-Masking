import json
import re
from typing import List, Dict, Tuple
from rapidfuzz import process, fuzz
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class NepaliFoodExtractor:

    def __init__(
        self,
        json_path: str = "sample_master.json",
        chroma_dir: str = "./chroma_db",
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            self.foods_data = json.load(f)["foods"]

        # Build clean mapping from aliases to food IDs
        self.alias_to_id = {}
        self.all_aliases = []

        for food in self.foods_data:
            food_id = food["id"]
            for alias in food.get("other_names", []):
                norm_alias = alias.lower().strip()
                self.alias_to_id[norm_alias] = food_id
                self.all_aliases.append(norm_alias)

        # Sort aliases by length descending so longer phrases ("kukhura ko masu") match before short ones ("masu")
        self.all_aliases.sort(key=len, reverse=True)

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.vectorstore = Chroma(
            persist_directory=chroma_dir, embedding_function=self.embedding_model
        )

    def extract_and_calculate(
        self, sentence: str, default_plate_gram: float = 150.0
    ):
        sentence_clean = sentence.lower().strip()
        matched_foods = {}

        # 1. Exact & Fuzzy matching across sentence substrings
        for alias in self.all_aliases:
            food_id = self.alias_to_id[alias]

            # Avoid matching duplicate food types unless already found
            if food_id in matched_foods:
                continue

            # Check if alias exists as a substring in the sentence or fuzzy matches a chunk
            if alias in sentence_clean or self._has_fuzzy_match(
                sentence_clean, alias
            ):
                grams = self._extract_quantity_for_alias(
                    sentence_clean, alias, default_plate_gram
                )
                matched_foods[food_id] = {
                    "matched_alias": alias,
                    "grams": grams,
                }

        # 2. ONLY fallback to ChromaDB if fuzzy/string search completely failed
        if not matched_foods:
            results = self.vectorstore.similarity_search_with_score(
                sentence_clean, k=1
            )
            if results and results[0][1] < 1.0:  # Similarity distance threshold
                doc, _ = results[0]
                food_id = doc.metadata["id"]
                matched_foods[food_id] = {
                    "matched_alias": doc.metadata["name"],
                    "grams": default_plate_gram,
                }

        # 3. Compute Macros
        total_nutrition = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
        itemized_results = []

        for food_id, info in matched_foods.items():
            food_obj = next(f for f in self.foods_data if f["id"] == food_id)
            grams = info["grams"]
            macros = food_obj["nutrition_per_gram"]

            item_cal = round(macros["calories"] * grams, 2)
            item_prot = round(macros["protein"] * grams, 2)
            item_carbs = round(macros["carbs"] * grams, 2)
            item_fat = round(macros["fat"] * grams, 2)

            total_nutrition["calories"] += item_cal
            total_nutrition["protein"] += item_prot
            total_nutrition["carbs"] += item_carbs
            total_nutrition["fat"] += item_fat

            itemized_results.append(
                {
                    "food_id": food_id,
                    "name": food_obj["name"],
                    "consumed_grams": grams,
                    "calories": item_cal,
                    "protein": item_prot,
                    "carbs": item_carbs,
                    "fat": item_fat,
                }
            )

        return {
            "itemized": itemized_results,
            "total_nutrition": {
                k: round(v, 2) for k, v in total_nutrition.items()
            },
        }

    def _has_fuzzy_match(self, text: str, alias: str) -> bool:
        """Check words in text against alias using ratio."""
        words = text.split()
        for word in words:
            if len(word) >= 3 and fuzz.ratio(word, alias) >= 82:
                return True
        return False

    def _extract_quantity_for_alias(
        self, sentence: str, alias: str, default_gram: float
    ) -> float:
        """Looks for numbers/units immediately preceding the alias."""
        # Find index of alias in sentence
        idx = sentence.find(alias)
        if idx == -1:
            idx = 0

        # Substring leading up to the item
        prefix = sentence[:idx].strip()
        words = prefix.split()

        # Look back up to 3 words before the item name
        lookback = " ".join(words[-3:]) if words else ""

        # Match exact grams e.g. "150 gram", "200g"
        gram_match = re.search(r"(\d+)\s*(gram|gm|g)", lookback)
        if gram_match:
            return float(gram_match.group(1))

        # Match plates e.g. "1 plate", "2 bowl"
        plate_match = re.search(r"(\d+)\s*(plate|katora|bowl)", lookback)
        if plate_match:
            return float(plate_match.group(1)) * default_gram

        # Fallback to standalone numbers preceding food e.g. "2 bhat" -> 2 plates
        num_match = re.search(r"(\d+)", lookback)
        if num_match:
            return float(num_match.group(1)) * default_gram

        return default_gram


# --- EXECUTION ---
if __name__ == "__main__":
    extractor = NepaliFoodExtractor()

    test_sentences = [
        "maile 1 plate dal ra 1 plate bhat khaye",
        "aaja bihan maile 150 gram kukhura ko masu ani 200 gram chiura khayeko thiye",
        "khana ma alu ra palung saag thiyo",
    ]

    for idx, sentence in enumerate(test_sentences, 1):
        print(f"\n================ TEST {idx} ================")
        print(f"INPUT SENTENCE: '{sentence}'")
        output = extractor.extract_and_calculate(sentence, default_plate_gram=150.0)

        print("\nIDENTIFIED FOOD ITEMS:")
        for item in output["itemized"]:
            print(
                f"  - [{item['food_id']}] {item['name']}: {item['consumed_grams']}g "
                f"({item['calories']} kcal, {item['protein']}g P, {item['carbs']}g C, {item['fat']}g F)"
            )

        print(f"\nTOTAL NUTRITION: {output['total_nutrition']}")