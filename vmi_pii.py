import streamlit as st
from google.cloud import dlp_v2
from google.cloud import storage
import os
import re

def initialize_dlp_client():
    return dlp_v2.DlpServiceClient()

def initialize_storage_client():
    return storage.Client()

def upload_to_gcs(bucket_name, file_name, file_content):
    storage_client = initialize_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(file_content)
    return f'gs://{bucket_name}/{file_name}'

def is_valid_project_id(project_id):
    pattern = r'^[a-z][a-z0-9-]{4,28}[a-z0-9]$'
    return re.match(pattern, project_id) is not None

def deidentify_content(project, content, info_types):
    dlp_client = initialize_dlp_client()

    # Construct the item
    item = {"value": content}

    # Construct the deidentify config
    deidentify_config = {
        "info_type_transformations": {
            "transformations": [
                {
                    "primitive_transformation": {
                        "character_mask_config": {
                            "masking_character": "*",
                            "number_to_mask": 100
                        }
                    }
                }
            ]
        }
    }

    # Construct the inspect config
    inspect_config = {
        "info_types": [{"name": info_type} for info_type in info_types]
    }

    # Call the API
    try:
        response = dlp_client.deidentify_content(
            request={
                "parent": f"projects/{project}",
                "deidentify_config": deidentify_config,
                "inspect_config": inspect_config,
                "item": item,
            }
        )
        return response.item.value
    except Exception as e:
        st.error(f"DLP API Error: {str(e)}")
        return None

def main():
    st.title("PII Masking Application using Google Cloud DLP")

    # Google Cloud settings
    project_id = st.text_input("Google Cloud Project ID")
    gcs_bucket = st.text_input("Google Cloud Storage Bucket Name")

    # Validate project ID
    if project_id and not is_valid_project_id(project_id):
        st.error("Invalid Project ID format. Please check your Project ID.")
        return

    # Ensure Google Cloud credentials are set
    if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        st.error("Google Cloud credentials not found. Please set the GOOGLE_APPLICATION_CREDENTIALS environment variable.")
        return

    # File upload
    uploaded_file = st.file_uploader("Choose a file", type=["txt", 'pdf'])

    if uploaded_file is not None and project_id and gcs_bucket:
        # Read file content
        content = uploaded_file.getvalue().decode("utf-8")

        # Display original content
        st.subheader("Original Content:")
        st.text_area("", content, height=200)

        # Define info types to look for
        info_types = [
            'PERSON_NAME', 'PHONE_NUMBER', 'EMAIL_ADDRESS', 'US_SOCIAL_SECURITY_NUMBER'
        ]

        # Deidentify content
        masked_content = deidentify_content(project_id, content, info_types)

        if masked_content:
            # Display masked content
            st.subheader("Masked Content:")
            st.text_area("", masked_content, height=200)

            try:
                # Upload masked content to GCS
                gcs_file_name = f"masked_{uploaded_file.name}"
                gcs_uri = upload_to_gcs(gcs_bucket, gcs_file_name, masked_content)

                # Provide download link
                st.success(f"Masked file uploaded to: {gcs_uri}")
                st.markdown(f"[Download masked file]({gcs_uri})")

                # Display the full masked content
                st.subheader("Full Masked Content:")
                st.text(masked_content)
            except Exception as e:
                st.error(f"Error uploading to Google Cloud Storage: {str(e)}")

if __name__ == "__main__":
    main()