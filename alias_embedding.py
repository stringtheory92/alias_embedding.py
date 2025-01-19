import csv
import hashlib
import json
import os

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from scipy.spatial.distance import cosine

import sys

csv.field_size_limit(sys.maxsize)

load_dotenv()

CSV_FILE = "adelaide_full.csv"
OUTPUT_FILE = "embeddings_output.json"
SOURCE_COLUMN = "source_column"
QUERY_SIGNATURE_HASH_COLUMN = "query_signature_hash"
RESULT_OUTPUT_FILE = "similarity_results.csv"
CHAR_LENGTH_THRESHOLD = 200  # Min char length filter
SIMILARITY_THRESHOLD = 0.80  # Similarity threshold filter

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_hash(text):
    return hashlib.md5(text.encode()).hexdigest()


def get_embedding(text):
    logger.info(f"Generating embedding for text: {text[:50]}...")
    response = client.embeddings.create(model="text-embedding-ada-002", input=text)
    return response.data[0].embedding


def load_cached_embeddings():
    """Load previously saved embeddings from OUTPUT_FILE."""
    if os.path.exists(OUTPUT_FILE):
        logger.info(f"Loading cached embeddings from {OUTPUT_FILE}")
        with open(OUTPUT_FILE, "r") as infile:
            return json.load(infile)
    else:
        logger.info(f"No cache found. Starting fresh.")

        return {}


def save_cached_embeddings(embeddings_dict):
    """Save embeddings to OUTPUT_FILE."""
    logger.info(f"Saving updated embeddings to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w") as outfile:
        json.dump(embeddings_dict, outfile)


def save_results_to_csv(similarities):
    """Save the similarity results to a CSV file."""
    logger.info(f"Saving similarity results to {RESULT_OUTPUT_FILE}")
    with open(RESULT_OUTPUT_FILE, "a", newline="", encoding="utf-8") as csvfile:  # Append mode
        fieldnames = ["source_column", "source_qsh", "matched_column", "matched_qsh", "similarity"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if csvfile.tell() == 0:
            writer.writeheader()

        for result in similarities:
            writer.writerow(
                {
                    "source_column": result["source_column"],
                    "source_qsh": result["source_qsh"],
                    "matched_column": result["matched_column"],
                    "matched_qsh": result["matched_qsh"],
                    "similarity": result["similarity"],
                }
            )


def process_csv():
    embeddings_dict = load_cached_embeddings()

    row_idx = 0
    with open(CSV_FILE, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        logger.info(f"Processing CSV file: {CSV_FILE}")
        for row in reader:
            source_text = row[SOURCE_COLUMN]
            query_signature_hash = row[QUERY_SIGNATURE_HASH_COLUMN]

            if len(source_text) < CHAR_LENGTH_THRESHOLD:
                logger.info(
                    f"Skipping source column: {source_text} (length below {CHAR_LENGTH_THRESHOLD})"
                )
                continue

            uniq_hash = generate_hash(source_text)

            if uniq_hash not in embeddings_dict:
                logger.info(f"Hash {uniq_hash} not found in cache. Generating embedding.")
                embedding = get_embedding(source_text)
                embeddings_dict[uniq_hash] = {
                    "embedding": embedding,
                    "text": source_text,
                    "query_signature_hash": query_signature_hash,
                }
            else:
                logger.info(f"Hash {uniq_hash} found in cache. Skipping embedding generation.")

            row_idx += 1

    save_cached_embeddings(embeddings_dict)


def compare_source_columns():
    """Iterates over the source columns and compares each one to the others using embeddings."""
    embeddings_dict = load_cached_embeddings()

    with open(CSV_FILE, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        logger.info(f"Processing and comparing source columns from CSV file: {CSV_FILE}")

        for row in reader:
            target_definition = row[SOURCE_COLUMN]
            target_qsh = row[QUERY_SIGNATURE_HASH_COLUMN]

            if len(target_definition) < CHAR_LENGTH_THRESHOLD:
                continue

            find_similar_embeddings_and_filter(target_definition.strip(), target_qsh)


def find_similar_embeddings_and_filter(target_definition, target_qsh, top_n=5):
    embeddings_dict = load_cached_embeddings()

    target_hash = generate_hash(target_definition)
    if target_hash not in embeddings_dict:
        target_embedding = get_embedding(target_definition)
    else:
        target_embedding = embeddings_dict[target_hash]["embedding"]

    similarities = []
    for hash, embedding in embeddings_dict.items():
        if hash != target_hash:
            similarity = 1 - cosine(np.array(target_embedding), np.array(embedding["embedding"]))
            if similarity >= SIMILARITY_THRESHOLD:
                similarities.append(
                    {
                        "source_column": target_definition,
                        "source_qsh": target_qsh,
                        "matched_column": embedding["text"],
                        "matched_qsh": embedding["query_signature_hash"],
                        "similarity": similarity,
                    }
                )

    similarities.sort(key=lambda x: x["similarity"], reverse=True)

    if similarities:
        logger.info(
            f"Found {len(similarities)} similar definitions with similarity >= {SIMILARITY_THRESHOLD}."
        )
        save_results_to_csv(similarities)

    return similarities[:top_n]


if __name__ == "__main__":
    # Step 1: Process CSV and generate embeddings if needed
    process_csv()

    # Step 2: Iterate over all source_column values in CSV_FILE and compare them
    compare_source_columns()
