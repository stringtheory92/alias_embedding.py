import csv
import os
import json
from collections import defaultdict
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

load_dotenv()

# Input/Output file paths
SIMILARITY_FILE = "similarity_results.csv"  # Your input CSV file
ANALYSIS_OUTPUT_FILE = "matched_analysis_enhanced.csv"  # File for analysis output

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Ensure result output file exists
if not os.path.exists(SIMILARITY_FILE):
    logger.error(f"Similarity results file {SIMILARITY_FILE} not found.")
    exit(1)


def count_tokens(text, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def chat_completion(user_prompt, system_prompt, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
    )
    return response.choices[0].message.content.strip()


def analyze_with_openai(source_column, matched_columns):
    """
    Perform analysis using OpenAI's gpt-4o-mini model by sending a prompt based on source_column and matched_columns.
    """

    system_prompt = """You are a specialized assistant tasked with analyzing SQL column definitions to help identify potential data fragmentation issues for a company. Fragmentation can occur when metrics are inconsistently labeled or derived differently across teams, even if they serve similar purposes. To achieve this, you should consider:

    1. **Aliases and Naming Conventions**: Aliases in SQL can indicate metric purposes across queries (e.g., AS unique_views, AS revenue) but should not be solely relied upon due to the risk of both false positives (different metrics with similar names) and false negatives (same metrics labeled differently across teams). Use aliases as hints rather than definitive indicators.

    2. **Intent Analysis (Query Logic)**: The structure of queries—including SELECT, GROUP BY, and WHERE clauses—provides insight into what the query is calculating. For example, a query using COUNT(DISTINCT user_id) likely aims to count unique users, while SUM(amount) may be focused on revenue. Intent is more indicative of a metric’s purpose than the alias alone.

    3. **Business Logic Mapping**: Metrics often vary by derivation across teams. For instance, "Revenue" might be calculated as gross sales by one team and as net sales by another. Align metrics to predefined templates to help recognize similar business objectives, even when SQL structures differ. **Classify identified differences based on likely business logic discrepancies**, such as distinguishing between "aggregated metrics" (e.g., SUM) vs. "distinct counts" (e.g., COUNT DISTINCT).

    4. **Query Metadata and Patterns**: Beyond aliases, examining metadata patterns can provide clues. Queries that source data from similar tables (e.g., user or order tables) and apply similar filtering conditions (e.g., by time or category) likely aim to calculate the same or similar metrics.

    5. **Deprioritize Cosmetic Differences**: Minor naming variations, such as different abbreviations, should be deprioritized if they don’t change metric intent. Focus instead on structural differences that impact business logic.

Using these principles, interpret the SQL columns provided in each task, assessing whether they indicate aligned or fragmented metric definitions.across teams, where different SQL queries may attempt to calculate the same business metric but do so inconsistently.

    You will be given:

    A source column definition, which serves as the baseline for comparison.
    A list of matched columns that have been identified as similar using cosine similarity.
    Your objective:

    Compare each matched column to the source column and to each other - the columns are from different queries, and unrelated except that they were found similar by cosine similarity.
    Identify and describe any inconsistencies or variations that could be signs of data fragmentation, such as:
    Variations in naming conventions or aliases.
    Differences in business logic (e.g., using gross sales vs. net sales to calculate revenue).
    Differences in filtering logic, aggregation functions, or grouping.
    Any other variations that suggest two teams may be trying to achieve the same goal but using different methods.
    Do not provide hypothetical scenarios. Focus only on the provided columns and make your analysis as specific as possible. If any of the columns are almost identical, except for minor differences that seem to come from different teams attempting to solve the same problem, highlight those differences.

    Output your analysis in a concise, structured manner, and ensure that any relevant insights are clear, reference the query code, directly related to the query code, and directly tied to solving the issue of fragmented data across teams.
    """

    # system_prompt = """
    # You are a specialized assistant tasked with analyzing SQL column definitions to help identify potential data fragmentation issues for a company. The company is trying to solve the problem of fragmented data across teams, where different SQL queries may attempt to calculate the same business metric but do so inconsistently.

    # You will be given:

    # A source column definition, which serves as the baseline for comparison.
    # A list of matched columns that have been identified as similar using cosine similarity.
    # Your objective:

    # Compare each matched column to the source column and to each other - the columns are from different queries, and unrelated except that they were found similar by cosine similarity.
    # Identify and describe any inconsistencies or variations that could be signs of data fragmentation, such as:
    # Variations in naming conventions or aliases.
    # Differences in business logic (e.g., using gross sales vs. net sales to calculate revenue).
    # Differences in filtering logic, aggregation functions, or grouping.
    # Any other variations that suggest two teams may be trying to achieve the same goal but using different methods.
    # Do not provide hypothetical scenarios. Focus only on the provided columns and make your analysis as specific as possible. If any of the columns are almost identical, except for minor differences that seem to come from different teams attempting to solve the same problem, highlight those differences.

    # Output your analysis in a concise, structured manner, and ensure that any relevant insights are clear, reference the query code, directly related to the query code, and directly tied to solving the issue of fragmented data across teams.
    # """

    user_prompt = f"""
    I would like to give you a high level overview of what my company is trying to achieve by parsing client metadata with sqlglot to solve data fragmentation issues:
    1. Aliases and Naming Conventions
    Aliases in SQL can help identify metrics with similar purposes across different queries (e.g., AS unique_views, AS revenue). However, relying solely on aliases can lead to both false positives (when different metrics are given similar names) and false negatives (when the same metric is labeled differently across teams). Therefore, while aliases are useful hints, they should not be the sole factor.
    2. Intent Analysis (Query Logic)
    The query structure, including SELECT, GROUP BY, and WHERE clauses, provides important insight into what the query is trying to calculate. For example:
    A query with COUNT(DISTINCT user_id) is likely trying to calculate a unique count of users, while another query using SUM(amount) may be focused on revenue. Even if both are named similarly (e.g., "engagement" or "total activity"), the underlying intent differs.
    The query's intent, rather than just its alias, should be considered in identifying metrics.
    3. Business Logic Mapping
    Associating metrics based on their purpose and logic is crucial. For instance, in business logic:
    "Revenue" might be derived differently across teams (e.g., one team uses gross sales, while another uses net sales).
    Twing could use predefined templates of business logic to recognize when two queries are intended to represent the same metric, despite different derivations. For example, "unique views" could have several variations but might still align with the same business metric goal.
    4. Query Metadata and Patterns
    Beyond aliases, looking at metadata patterns can help. For example, if two queries:
    Pull data from the same key tables (e.g., a user table or an order table),
    Apply similar filtering (e.g., filtering by time window, product category), They likely aim to compute the same metric even if the exact SQL structure differs.

    next I want to provide you with a group of column definitions from various queries from one of our client companies. The first column definition is the source column all other columns in all the query data we have for them was tested against with embeddings and cosine similarity. 
    The remainder of column definitions are the ones that were found to be greater than 80% similar. I would like you to evaluate to what degree the columns may or may not be notably similar or the same as the first column or each other, and if there are any where they are almost the same but only different in a way that could be born of two people from two teams trying to do the same thing but it's being done inconsistently (data fragmentation), please tell me. stand by for the first set of column definitions!

    Source Column: {source_column}
    
    Matched Columns: {', '.join(matched_columns)}
    
    -- Please keep in mind that I don't need hypothetical examples of how fragmentation 'could' occur. I want any examples of it perhaps happening based on any found inconsistencies in the queries.
    """

    # user_prompt = f"""
    # I would like to give you a high level overview of what my company is trying to achieve by parsing client metadata with sqlglot to solve data fragmentation issues:
    # 1. Aliases and Naming Conventions
    # Aliases in SQL can help identify metrics with similar purposes across different queries (e.g., AS unique_views, AS revenue). However, relying solely on aliases can lead to both false positives (when different metrics are given similar names) and false negatives (when the same metric is labeled differently across teams). Therefore, while aliases are useful hints, they should not be the sole factor.
    # 2. Intent Analysis (Query Logic)
    # The query structure, including SELECT, GROUP BY, and WHERE clauses, provides important insight into what the query is trying to calculate. For example:
    # A query with COUNT(DISTINCT user_id) is likely trying to calculate a unique count of users, while another query using SUM(amount) may be focused on revenue. Even if both are named similarly (e.g., "engagement" or "total activity"), the underlying intent differs.
    # The query's intent, rather than just its alias, should be considered in identifying metrics.
    # 3. Business Logic Mapping
    # Associating metrics based on their purpose and logic is crucial. For instance, in business logic:
    # "Revenue" might be derived differently across teams (e.g., one team uses gross sales, while another uses net sales).
    # Twing could use predefined templates of business logic to recognize when two queries are intended to represent the same metric, despite different derivations. For example, "unique views" could have several variations but might still align with the same business metric goal.
    # 4. Query Metadata and Patterns
    # Beyond aliases, looking at metadata patterns can help. For example, if two queries:
    # Pull data from the same key tables (e.g., a user table or an order table),
    # Apply similar filtering (e.g., filtering by time window, product category), They likely aim to compute the same metric even if the exact SQL structure differs.

    # next I want to provide you with a group of column definitions from various queries from one of our client companies. The first column definition is the source column all other columns in all the query data we have for them was tested against with embeddings and cosine similarity.
    # The remainder of column definitions are the ones that were found to be greater than 80% similar. I would like you to evaluate to what degree the columns may or may not be notably similar or the same as the first column or each other, and if there are any where they are almost the same but only different in a way that could be born of two people from two teams trying to do the same thing but it's being done inconsistently (data fragmentation), please tell me. stand by for the first set of column definitions!

    # Source Column: {source_column}

    # Matched Columns: {', '.join(matched_columns)}

    # -- Please keep in mind that I don't need hypothetical examples of how fragmentation 'could' occur. I want any examples of it perhaps happening based on any found inconsistencies in the queries.
    # """

    system_tokens = count_tokens(system_prompt)
    user_tokens = count_tokens(user_prompt)

    total_tokens = system_tokens + user_tokens

    if total_tokens > 128000:  # Adjust threshold based on the actual model's limit
        logger.warning("Token limit exceeded! Trimming content.")
        user_prompt = user_prompt[: int(128000 - system_tokens)]

    try:
        response = chat_completion(user_prompt, system_prompt)
        # Extract and return the completion text
        return response

    except Exception as e:
        logger.error(f"Failed to get a response from OpenAI: {e}")
        return "Error during analysis"


def process_similarity_results():
    """
    Process the similarity results, call the OpenAI API for analysis, and save the output.
    """
    logger.info(f"Processing similarity results from {SIMILARITY_FILE}")

    # Dictionary to group by source_column
    grouped_data = defaultdict(list)

    # Read similarity_results.csv and group by source_column
    with open(SIMILARITY_FILE, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            source_column = row["source_column"]
            matched_column = row["matched_column"]
            query_signature_hash = row["source_qsh"]

            # Append matched_column for the corresponding source_column
            grouped_data[source_column].append((query_signature_hash, matched_column))

    # Now perform analysis for each unique source_column
    logger.info(f"Found {len(grouped_data)} unique source columns for analysis.")
    logger.info(f"{grouped_data}")

    # Open output file for writing the results of the analysis
    with open(ANALYSIS_OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["query_signature_hash", "source_column", "matched_analysis"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Loop through each source_column and its matched_columns
        for source_column, matched_list in grouped_data.items():
            query_signature_hash = matched_list[0][0]  # Get the QSH from the first match
            matched_columns = [
                matched[1] for matched in matched_list
            ]  # Extract all matched columns

            logger.info(f"Analyzing source column: {source_column}")

            # Perform the OpenAI analysis
            analysis = analyze_with_openai(source_column, matched_columns)
            logger.info(f"Analysis: {analysis}")

            # Write the result to the output CSV
            writer.writerow(
                {
                    "query_signature_hash": query_signature_hash,
                    "source_column": source_column,
                    "matched_analysis": analysis,
                }
            )


if __name__ == "__main__":
    # Process similarity results and perform OpenAI analysis
    process_similarity_results()
