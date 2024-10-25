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

# Increase the CSV field size limit
csv.field_size_limit(sys.maxsize)

# Load environment variables from .env file
load_dotenv()

CSV_FILE = "adelaide_full.csv"
OUTPUT_FILE = "embeddings_output.json"
SOURCE_COLUMN = "source_column"
QUERY_SIGNATURE_HASH_COLUMN = "query_signature_hash"
RESULT_OUTPUT_FILE = "similarity_results.csv"
CHAR_LENGTH_THRESHOLD = 200  # Set your desired minimum character length
SIMILARITY_THRESHOLD = 0.80  # Similarity threshold for storing results

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_hash(text):
    return hashlib.md5(text.encode()).hexdigest()


def get_embedding(text):
    logger.info(
        f"Generating embedding for text: {text[:50]}..."
    )  # Show first 50 chars for readability
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
        # Create an empty dictionary to start from if the file doesn't exist
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

        # Write header only if the file is empty
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

            # Ensure source column text meets the character length threshold
            if len(source_text) < CHAR_LENGTH_THRESHOLD:
                logger.info(
                    f"Skipping source column: {source_text} (length below {CHAR_LENGTH_THRESHOLD})"
                )
                continue

            uniq_hash = generate_hash(source_text)

            # Check if embedding is already cached
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

        # Iterate through all rows in the CSV file
        for row in reader:
            target_definition = row[SOURCE_COLUMN]
            target_qsh = row[QUERY_SIGNATURE_HASH_COLUMN]

            # Skip source columns that don't meet the length requirement
            if len(target_definition) < CHAR_LENGTH_THRESHOLD:
                continue

            # Compare this source column to all the others
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

    # Sort similarities in descending order and limit to top_n results
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


# import csv
# import hashlib
# import json
# import os

# import numpy as np
# from dotenv import load_dotenv
# from loguru import logger
# from openai import OpenAI
# from scipy.spatial.distance import cosine

# import csv
# import sys

# # Increase the CSV field size limit
# csv.field_size_limit(sys.maxsize)

# # Load environment variables from .env file
# load_dotenv()

# CSV_FILE = "adelaide_full.csv"
# OUTPUT_FILE = "embeddings_output.json"
# SOURCE_COLUMN = "source_column"  # Replace with the actual column name

# # Initialize OpenAI client
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# def generate_hash(text):
#     return hashlib.md5(text.encode()).hexdigest()


# def get_embedding(text):
#     logger.info(f"Generating embedding for text hash: {generate_hash(text)}")
#     response = client.embeddings.create(model="text-embedding-3-small", input=text)
#     return response.data[0].embedding


# def load_cached_embeddings():
#     """Load previously saved embeddings from OUTPUT_FILE."""
#     if os.path.exists(OUTPUT_FILE):
#         logger.info(f"Loading cached embeddings from {OUTPUT_FILE}")
#         with open(OUTPUT_FILE, "r") as infile:
#             return json.load(infile)
#     else:
#         logger.info(f"No cache found. Starting fresh.")
#     return {}


# def process_csv():
#     embeddings_dict = load_cached_embeddings()

#     row_idx = 0
#     with open(CSV_FILE, "r", newline="", encoding="utf-8") as csvfile:
#         reader = csv.DictReader(csvfile)
#         logger.info(f"Processing CSV file: {CSV_FILE}")
#         for row in reader:
#             source_text = row[SOURCE_COLUMN]
#             uniq_hash = generate_hash(source_text)

#             # Check if embedding is already cached
#             if uniq_hash not in embeddings_dict:
#                 logger.info(f"Hash {uniq_hash} not found in cache. Generating embedding.")
#                 embedding = get_embedding(source_text)
#                 embeddings_dict[uniq_hash] = {
#                     "embedding": embedding,
#                     "text": source_text,
#                 }
#             else:
#                 logger.info(f"Hash {uniq_hash} found in cache. Skipping embedding generation.")

#             row_idx += 1
#             if row_idx > 10000:
#                 logger.info(f"Stopping processing after {row_idx} rows.")
#                 break

#     logger.info(f"Saving updated embeddings to {OUTPUT_FILE}")
#     with open(OUTPUT_FILE, "w") as outfile:
#         json.dump(embeddings_dict, outfile)


# def find_similar_embeddings(target_definition, top_n=5):
#     with open(OUTPUT_FILE, "r") as infile:
#         embeddings_dict = json.load(infile)

#     target_hash = generate_hash(target_definition)
#     if target_hash not in embeddings_dict:
#         target_embedding = get_embedding(target_definition)
#     else:
#         target_embedding = embeddings_dict[target_hash]["embedding"]

#     similarities = []
#     for hash, embedding in embeddings_dict.items():
#         if hash != target_hash:
#             similarity = 1 - cosine(target_embedding, embedding["embedding"])
#             similarities.append((hash, similarity))

#     similarities.sort(key=lambda x: x[1], reverse=True)

#     print(f"Top {top_n} similar definitions to '{target_definition}':")
#     for hash, similarity in similarities[:top_n]:
#         print(f"Hash: {hash}, Similarity: {similarity:.4f}")
#         print(embeddings_dict[hash]["text"])

#     return similarities[:top_n]


# if __name__ == "__main__":
#     # process_csv()
#     # logger.info(f"Embeddings saved to {OUTPUT_FILE}")

#     target_definition = """
#     SELECT adelaide_campaign_id , campaign_name_raw , ad_group_name_raw AS ad_set_name_raw , ad_name_raw , campaign_id , ad_id AS creative_id , inferred_duration , COALESCE(inferred_placement, 'unknown') as inferred_placement , CASE WHEN inferred_ad_type = 'nan' then NULL else inferred_ad_type END AS inferred_ad_format , case when cast(datediff(day, date, getdate()) as int) <0 then datediff(day, to_date(date, 'YYYY-DD-MM'), getdate()) else datediff(day, date, getdate()) end as days_old , SUM(GREATEST(impressions, earned_impressions, paid_impressions)) AS paid_impressions , SUM(GREATEST(paid_video_views_25prct, earned_video_views_25prct, total_video_views_25prct, video_views_25prct)) AS video_plays_at_25pct , SUM(GREATEST(paid_video_views_50prct, earned_video_views_50prct, total_video_views_50prct, video_views_50prct)) AS video_plays_at_50pct , SUM(GREATEST(paid_video_views_75prct, earned_video_views_75prct, total_video_views_75prct, video_views_75prct)) AS video_plays_at_75pct , SUM(GREATEST(paid_video_views_100prct, earned_video_views_100prct, total_video_views_100prct, video_views_100prct)) AS video_completions , SUM(spend) AS amount_spent FROM walled_gardens.pinterest WHERE GREATEST(paid_video_views_25prct, earned_video_views_25prct, total_video_views_25prct) > 0 AND GREATEST(impressions, earned_impressions, paid_impressions) > 0 AND days_old <= 90 AND days_old >= 0 AND date <= '2024-08-10' GROUP BY 1,2,3,4,5,6,7,8,9,10
#     """
#     # target_definition = """
#     # TS_OR_DS_DIFF(CURRENT_TIMESTAMP(), "a"."txn_start", SECOND) / 86400 || ' days ' || TS_OR_DS_DIFF(CURRENT_TIMESTAMP(), "a"."txn_start", SECOND) % 86400 / 3600 || ' hrs ' || TS_OR_DS_DIFF(CURRENT_TIMESTAMP(), "a"."txn_start", SECOND) % 3600 / 60 || ' mins ' || TS_OR_DS_DIFF(CURRENT_TIMESTAMP(), "a"."txn_start", SECOND) % 60 ||  secs
#     # """
#     # target_definition = """
#     # CASE WHEN "omnichannel_domain_report"."adelaide_data_source" IN ('adelaide_external_reporting', 'adelaide_video_rollup') THEN 'OLV' WHEN "omnichannel_domain_report"."adelaide_data_source" IN ('audio_pixels_podcast', 'audio_pixels_podcast_offline') THEN 'Podcast' WHEN "omnichannel_domain_report"."adelaide_data_source" IN ('audio_pixels_streaming_audio', 'streaming_audio_offline') THEN 'Streaming Audio' WHEN "omnichannel_domain_report"."adelaide_data_source" LIKE '%walled_garden%' THEN INITCAP("omnichannel_domain_report"."destination_source") WHEN "omnichannel_domain_report"."adelaide_data_source" IN ('adelaide_reporting_ctv', 'adelaide_rollup_ctv_offline', 'ctv_teads_log', 'external_reporting_ctv') THEN 'CTV' WHEN "omnichannel_domain_report"."adelaide_data_source" IN ('external_native', 'native_pixels', 'native_pixels_temp') THEN 'Native' WHEN "omnichannel_domain_report"."adelaide_data_source" = 'linear' THEN 'Linear TV' WHEN "omnichannel_domain_report"."adelaide_data_source" = 'cinema_au' THEN 'Cinema' WHEN "omnichannel_domain_report"."destination_source" = 'TrueX' THEN 'OLV' WHEN "omnichannel_domain_report"."adelaide_data_source" IN ('adelaide_rollup_auv3', 'external_display_audit') AND "aam__metadata_snippets_narrow"."is_video" = TRUE THEN 'OLV' WHEN "omnichannel_domain_report"."adelaide_data_source" IN ('adelaide_rollup_auv3', 'external_display_audit') THEN 'Display' ELSE NULL END
#     # """
#     find_similar_embeddings(target_definition.strip())


# import csv
# import hashlib
# import json
# import os

# import numpy as np
# from dotenv import load_dotenv
# from openai import OpenAI
# from scipy.spatial.distance import cosine
# from loguru import logger


# import csv
# import sys

# # Increase the CSV field size limit
# csv.field_size_limit(sys.maxsize)

# # Load environment variables from .env file
# load_dotenv()

# CSV_FILE = "adelaide_full.csv"
# OUTPUT_FILE = "embeddings_output.json"
# SOURCE_COLUMN = "source_column"  # Replace with the actual column name

# # Initialize OpenAI client
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# def generate_hash(text):
#     return hashlib.md5(text.encode()).hexdigest()


# def get_embedding(text):
#     response = client.embeddings.create(model="text-embedding-ada-002", input=text)
#     return response.data[0].embedding


# def process_csv():
#     embeddings_dict = {}

#     row_idx = 0
#     with open(CSV_FILE, "r", newline="", encoding="utf-8") as csvfile:
#         reader = csv.DictReader(csvfile)
#         for row in reader:
#             source_text = row[SOURCE_COLUMN]
#             uniq_hash = generate_hash(source_text)

#             if uniq_hash not in embeddings_dict:
#                 logger.info(f"Generating new embedding for {source_text}")
#                 embedding = get_embedding(source_text)
#                 embeddings_dict[uniq_hash] = {
#                     "embedding": embedding,
#                     "text": source_text,
#                 }
#             else:
#                 logger.info(f"Found cached embedding for {source_text}")
#             row_idx += 1
#             if row_idx > 100:
#                 break

#     with open(OUTPUT_FILE, "w") as outfile:
#         json.dump(embeddings_dict, outfile)


# def find_similar_embeddings(target_definition, top_n=5):
#     with open(OUTPUT_FILE, "r") as infile:
#         embeddings_dict = json.load(infile)

#     target_hash = generate_hash(target_definition)
#     if target_hash not in embeddings_dict:
#         target_embedding = get_embedding(target_definition)
#     else:
#         target_embedding = embeddings_dict[target_hash]["embedding"]

#     similarities = []
#     for hash, embedding in embeddings_dict.items():
#         if hash != target_hash:
#             similarity = 1 - cosine(target_embedding, embedding["embedding"])
#             similarities.append((hash, similarity))

#     similarities.sort(key=lambda x: x[1], reverse=True)

#     print(f"Top {top_n} similar definitions to '{target_definition}':")
#     for hash, similarity in similarities[:top_n]:
#         print(f"Hash: {hash}, Similarity: {similarity:.4f}")
#         print(embeddings_dict[hash]["text"])

#     return similarities[:top_n]


# if __name__ == "__main__":
#     # Uncomment the following lines if you need to regenerate embeddings
#     process_csv()
#     print(f"Embeddings saved to {OUTPUT_FILE}")

#     # Example usage of find_similar_embeddings
#     target_definition = """
#     CASE WHEN "omnichannel_domain_report"."adelaide_data_source" IN ('adelaide_external_reporting', 'adelaide_video_rollup') THEN 'OLV' WHEN "omnichannel_domain_report"."adelaide_data_source" IN ('audio_pixels_podcast', 'audio_pixels_podcast_offline') THEN 'Podcast' WHEN "omnichannel_domain_report"."adelaide_data_source" IN ('audio_pixels_streaming_audio', 'streaming_audio_offline') THEN 'Streaming Audio' WHEN "omnichannel_domain_report"."adelaide_data_source" LIKE '%walled_garden%' THEN INITCAP("omnichannel_domain_report"."destination_source") WHEN "omnichannel_domain_report"."adelaide_data_source" IN ('adelaide_reporting_ctv', 'adelaide_rollup_ctv_offline', 'ctv_teads_log', 'external_reporting_ctv') THEN 'CTV' WHEN "omnichannel_domain_report"."adelaide_data_source" IN ('external_native', 'native_pixels', 'native_pixels_temp') THEN 'Native' WHEN "omnichannel_domain_report"."adelaide_data_source" = 'linear' THEN 'Linear TV' WHEN "omnichannel_domain_report"."adelaide_data_source" = 'cinema_au' THEN 'Cinema' WHEN "omnichannel_domain_report"."destination_source" = 'TrueX' THEN 'OLV' WHEN "omnichannel_domain_report"."adelaide_data_source" IN ('adelaide_rollup_auv3', 'external_display_audit') AND "aam__metadata_snippets_narrow"."is_video" = TRUE THEN 'OLV' WHEN "omnichannel_domain_report"."adelaide_data_source" IN ('adelaide_rollup_auv3', 'external_display_audit') THEN 'Display' ELSE NULL END
#     """
#     find_similar_embeddings(target_definition.strip())
