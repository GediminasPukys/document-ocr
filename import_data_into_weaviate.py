import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from time import sleep
from openai import OpenAI

client = OpenAI()

# Add a warning about OpenAI library version
st.warning(
    "This script requires OpenAI library version 0.28. If you encounter errors, please run: pip install openai==0.28")


# Load and clean data
@st.cache_data
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)

    # Replace inf and -inf with a large number instead of NaN
    df = df.replace([np.inf, -np.inf], 1e10)

    # Convert POPULARITY to int, replacing NaN with 0
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


def manipulate_data(df):
    df_manipulated = df.copy()
    df_manipulated['COMBINED_NAME'] = df_manipulated.apply(concatenate_fields, axis=1)
    df_manipulated['DESCRIPTION'] = df_manipulated.apply(
        lambda row: row['DESCRIPTION'] + f" URL: {row['SHORT_NAME']}" if is_url(row['SHORT_NAME']) else row[
            'DESCRIPTION'], axis=1)
    df_manipulated['DESCRIPTION'] = df_manipulated.apply(
        lambda row: row['DESCRIPTION'] + f" URL: {row['LONG_NAME']}" if is_url(row['LONG_NAME']) else row[
            'DESCRIPTION'], axis=1)
    return df_manipulated


def enrich_selected_description(df, selected_index):
    df_enriched = df.copy()
    df_enriched.at[selected_index, 'DESCRIPTION'] = enrich_description_with_gpt(df_enriched.loc[selected_index])
    return df_enriched


def main():
    st.title("Service Description Data Manipulation and Enrichment Viewer (Using OpenAI GPT)")

    file_path = "CC_QUICKSTART_CORTEX_DOCS_DATA_SERVICES.csv"
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return

    # Load and manipulate data
    df_original = load_and_clean_data(file_path)
    df_manipulated = manipulate_data(df_original)

    # Print out initial file
    st.subheader("File Contents")
    st.dataframe(df_manipulated)

    # Select line to enrich
    selected_index = st.number_input("Select line number to enrich (0-based index)",
                                     min_value=0,
                                     max_value=len(df_manipulated) - 1,
                                     value=0)

    if st.button("Enrich Selected Line"):
        with st.spinner("Enriching description with GPT..."):
            df_enriched = enrich_selected_description(df_manipulated, selected_index)

        st.subheader(f"Enriched Data for Line {selected_index}")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Original Data**")
            st.write(f"SHORT_NAME: {df_original.loc[selected_index, 'SHORT_NAME']}")
            st.write(f"LONG_NAME: {df_original.loc[selected_index, 'LONG_NAME']}")
            st.write(f"PROVIDER_NAMES: {df_original.loc[selected_index, 'PROVIDER_NAMES']}")
            st.write(f"DESCRIPTION: {df_original.loc[selected_index, 'DESCRIPTION']}")
            st.write(f"CATEGORIES: {df_original.loc[selected_index, 'CATEGORIES']}")
            st.write(f"LIFE_EVENTS: {df_original.loc[selected_index, 'LIFE_EVENTS']}")

        with col2:
            st.write("**Enriched Data**")
            st.write(f"COMBINED_NAME: {df_enriched.loc[selected_index, 'COMBINED_NAME']}")
            st.write(f"ENRICHED DESCRIPTION: {df_enriched.loc[selected_index, 'DESCRIPTION']}")
            st.write(f"CATEGORIES: {df_enriched.loc[selected_index, 'CATEGORIES']}")
            st.write(f"LIFE_EVENTS: {df_enriched.loc[selected_index, 'LIFE_EVENTS']}")

        # Highlight changes
        if df_original.loc[selected_index, 'DESCRIPTION'] != df_enriched.loc[selected_index, 'DESCRIPTION']:
            st.info("The DESCRIPTION field has been enriched by GPT.")
        if is_url(df_original.loc[selected_index, 'SHORT_NAME']) or is_url(
                df_original.loc[selected_index, 'LONG_NAME']):
            st.info("A URL was detected and added to the DESCRIPTION.")


if __name__ == "__main__":
    main()