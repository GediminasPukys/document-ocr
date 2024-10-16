import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import json
from time import sleep
from openai import OpenAI
import weaviate
from tqdm import tqdm

client = OpenAI()

# Weaviate client setup
wcd_api_key = os.environ.get("WCS_API_KEY")
wcd_url = 'https://doryjgsbqxaoy4efqu6iwq.c0.europe-west3.gcp.weaviate.cloud'

weaviate_client = weaviate.Client(
    url=wcd_url,
    auth_client_secret=weaviate.AuthApiKey(api_key=wcd_api_key)
)

# Add a warning about OpenAI library version
st.warning(
    "This script requires OpenAI library version 0.28. If you encounter errors, please run: pip install openai==0.28")


@st.cache_data
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    df = df.replace([np.inf, -np.inf], 1e10)
    df['POPULARITY'] = df['POPULARITY'].fillna(0).astype(int)
    text_columns = ['LONG_NAME', 'SHORT_NAME', 'DESCRIPTION', 'SHORT_DESCRIPTION', 'KEYWORDS', 'CATEGORIES',
                    'LIFE_EVENTS', 'PROVIDER_NAMES']
    for col in text_columns:
        df[col] = df[col].fillna('').astype(str).str.strip()
    return df


def concatenate_fields(row):
    return f"{row['SHORT_NAME']} {row['LONG_NAME']} {row['PROVIDER_NAMES']}".strip()


def is_url(text):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return bool(url_pattern.search(text))


def enrich_description_with_gpt(row):
    prompt = f"""
    Given the following information about a Lithuanian public service provider, please specify and enrich which services the provider is offering:

    Service Name: {row['COMBINED_NAME']}
    Current Description: {row['DESCRIPTION']}
    Categories: {row['CATEGORIES']}
    Life Events: {row['LIFE_EVENTS']}
    Keywords: {row['KEYWORDS']}

    Please provide a detailed description of the services offered, expanding on the current description and incorporating relevant information from the categories, life events, and keywords. Return results translated into Lithuanian language.
    """

    st.text("Prompt for GPT enrichment:")
    st.code(prompt, language="text")

    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that specializes in describing Lithuanian public services."},
                {"role": "user", "content": prompt}
            ]
        )
        enriched_description = response.choices[0].message.content
        return f"{row['DESCRIPTION']} {enriched_description}"
    except Exception as e:
        st.error(f"Error in GPT enrichment: {str(e)}")
        return row['DESCRIPTION']


def enrich_and_save(df, start_index=0, output_dir="enriched_services"):
    os.makedirs(output_dir, exist_ok=True)

    progress_bar = st.progress(0)
    status_text = st.empty()

    for index, row in tqdm(df.iloc[start_index:].iterrows(), total=len(df) - start_index):
        try:
            # Enrich the description
            enriched_description = enrich_description_with_gpt(row)

            # Prepare the data object
            data_object = {
                "iD": str(row["ID"]),
                "LONG_NAME": row["LONG_NAME"],
                "SHORT_NAME": row["SHORT_NAME"],
                "DESCRIPTION": row["DESCRIPTION"],
                "SHORT_DESCRIPTION": row["SHORT_DESCRIPTION"],
                "KEYWORDS": row["KEYWORDS"],
                "CATEGORIES": row["CATEGORIES"],
                "LIFE_EVENTS": row["LIFE_EVENTS"],
                "PROVIDER_NAMES": row["PROVIDER_NAMES"],
                "POPULARITY": int(row["POPULARITY"]),
                "COMBINED_NAME": row["COMBINED_NAME"],
                "ENRICHED_DESCRIPTION": enriched_description
            }

            # Save to file
            file_path = os.path.join(output_dir, f"{row['ID']}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data_object, f, ensure_ascii=False, indent=4)

            st.write(f"Saved enriched data for ID {row['ID']}")

        except Exception as e:
            st.error(f"Error processing row {index} (ID: {row['ID']}): {str(e)}")
            continue

        # Update progress
        progress = (index - start_index + 1) / (len(df) - start_index)
        progress_bar.progress(progress)
        status_text.text(f"Processed {index - start_index + 1} out of {len(df) - start_index} rows")

        # Save progress every 10 rows
        if (index - start_index + 1) % 10 == 0:
            st.session_state.last_processed_index = index

    status_text.text("Enrichment and saving completed!")


def create_weaviate_schema():
    schema = {
        "class": "ServiceDescription",
        "vectorizer": "none",
        "properties": [
            {"name": "iD", "dataType": ["string"]},
            {"name": "LONG_NAME", "dataType": ["text"]},
            {"name": "SHORT_NAME", "dataType": ["text"]},
            {"name": "DESCRIPTION", "dataType": ["text"]},
            {"name": "SHORT_DESCRIPTION", "dataType": ["text"]},
            {"name": "KEYWORDS", "dataType": ["text"]},
            {"name": "CATEGORIES", "dataType": ["text"]},
            {"name": "LIFE_EVENTS", "dataType": ["text"]},
            {"name": "PROVIDER_NAMES", "dataType": ["text"]},
            {"name": "POPULARITY", "dataType": ["int"]},
            {"name": "COMBINED_NAME", "dataType": ["text"]},
            {"name": "ENRICHED_DESCRIPTION", "dataType": ["text"]}
        ]
    }
    weaviate_client.schema.create_class(schema)


def upload_to_weaviate(input_dir="enriched_services"):
    if not weaviate_client.schema.exists("ServiceDescription"):
        create_weaviate_schema()

    progress_bar = st.progress(0)
    status_text = st.empty()

    file_list = os.listdir(input_dir)
    for i, filename in enumerate(file_list):
        if filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data_object = json.load(f)

                # Upload to Weaviate
                weaviate_client.data_object.create(
                    class_name="ServiceDescription",
                    data_object=data_object
                )
                st.write(f"Uploaded record for ID {data_object['iD']}")

            except Exception as e:
                st.error(f"Error uploading file {filename}: {str(e)}")
                continue

        # Update progress
        progress = (i + 1) / len(file_list)
        progress_bar.progress(progress)
        status_text.text(f"Processed {i + 1} out of {len(file_list)} files")

    status_text.text("Upload to Weaviate completed!")


def main():
    st.title("Service Description Data Enrichment and Weaviate Upload")

    file_path = "CC_QUICKSTART_CORTEX_DOCS_DATA_SERVICES.csv"
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return

    # Load and manipulate data
    df_original = load_and_clean_data(file_path)
    df_manipulated = df_original.copy()
    df_manipulated['COMBINED_NAME'] = df_manipulated.apply(concatenate_fields, axis=1)

    # Display the number of rows
    st.info(f"Total number of rows: {len(df_manipulated)}")

    # Initialize session state for last processed index
    if 'last_processed_index' not in st.session_state:
        st.session_state.last_processed_index = 0

    # Enrich and save data
    if st.button("Start/Resume Enrichment and Saving"):
        enrich_and_save(df_manipulated, st.session_state.last_processed_index)

    # Upload to Weaviate
    if st.button("Upload Enriched Data to Weaviate"):
        upload_to_weaviate()

    # Display last processed index
    st.write(f"Last processed index: {st.session_state.last_processed_index}")


if __name__ == "__main__":
    main()