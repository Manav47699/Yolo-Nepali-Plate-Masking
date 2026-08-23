import json  # Standard JSON handling
import re  # Regex for text processing
from rapidfuzz import fuzz  # Fuzzy text matching


class SimplifiedFoodExtractor:

    def __init__(self, json_path: str = "sample_master.json"):  # Class constructor
        with open(json_path, "r", encoding="utf-8") as f:  # Open JSON database file
            self.foods_data = json.load(f)["foods"]  # Parse food list

        self.alias_to_food = {}  # Direct alias lookup mapping
        self.all_aliases = []  # List of all searchable aliases

        for food in self.foods_data:  # Loop through database foods
            for alias in food.get("other_names", []):  # Loop through food aliases
                norm_alias = alias.lower().strip()  # Normalize alias to lowercase
                self.alias_to_food[norm_alias] = food[
                    "name"
                ]  # Map alias to primary name
                self.all_aliases.append(norm_alias)  # Add alias to master list

        self.all_aliases.sort(
            key=len, reverse=True
        )  # Sort aliases by length (longest first)

    def extract_foods(self, sentence: str) -> list:  # Extract food names
        sentence_clean = sentence.lower().strip()  # Normalize input string
        extracted_items = []  # Result storage list

        for alias in self.all_aliases:  # Iterate through sorted aliases
            food_name = self.alias_to_food[alias]  # Retrieve food name

            if food_name in extracted_items:  # Avoid duplicate food extractions
                continue

            if alias in sentence_clean or self._fuzzy_match(
                sentence_clean, alias
            ):  # Substring/fuzzy check
                extracted_items.append(food_name)  # Append match to results

        return extracted_items  # Return extracted food list

    def _fuzzy_match(
        self, text: str, alias: str
    ) -> bool:  # Helper for slight spelling variations
        return any(
            fuzz.ratio(word, alias) >= 82 for word in text.split() if len(word) >= 3
        )  # Check word similarity


# --- EXAMPLE RUN ---
if __name__ == "__main__":
    extractor = SimplifiedFoodExtractor()  # Initialize extractor instance
    test_sentence = (
        "maile 1 plate dal ra bhat khaye"  # Input sentence to test
    )
    detected_foods = extractor.extract_foods(
        test_sentence
    )  # Process extraction

    print(f"Sentence: {test_sentence}")  # Print input text
    print(f"Detected Foods: {detected_foods}")  # Output: ['dal', 'bhat']