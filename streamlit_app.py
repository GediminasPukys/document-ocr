import streamlit as st
import requests
import time
import json

# API configuration
API_URL = 'https://api.cloud.llamaindex.ai/api/v1/parsing/upload'
API_KEY = 'llx-kHVYZKk7lHFXXL7QQ4S4ZPdvFqs1MypudZUYF4Jv6Jds7YTx'  # Consider using st.secrets for API keys

headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {API_KEY}',
}

# Streamlit UI
st.title("📄 Document Parsing")
st.write("Upload a document below for parsing!")

uploaded_file = st.file_uploader("Upload a document (.pdf)", type=["pdf"])

if uploaded_file:
    st.write("File uploaded successfully!")

    # Prepare the file for API request
    file_bytes = uploaded_file.getvalue()
    files = {
        'language': (None, 'lt'),
        'parsing_instruction': (None, 'File is an invoice. Parse invoice number, date, sender and receiver name, addresses, vat number, sum. Return only json with parsed key value pairs'),
        'file': (uploaded_file.name, file_bytes, 'application/pdf'),
        'bounding_box': (None, '0,0,0,0'),
    }

    # Send request to API
    if st.button("Parse Document"):
        with st.spinner("Uploading and parsing the document..."):
            try:
                response = requests.post(API_URL, headers=headers, files=files)
                response.raise_for_status()  # Raise an exception for bad status codes

                # Get job ID
                job_id = response.json()["id"]
                st.success(f"Document uploaded! Job ID: {job_id}")

                # Check job status
                status = "pending"
                status_placeholder = st.empty()
                while status != "SUCCESS":
                    response_job_status = requests.get(
                        f'https://api.cloud.llamaindex.ai/api/v1/parsing/job/{job_id}',
                        headers=headers
                    )
                    response_job_status.raise_for_status()
                    status = response_job_status.json()["status"]
                    status_placeholder.text(f"Job status: {status}")
                    if status != "SUCCESS":
                        time.sleep(5)  # Wait for 5 seconds before checking again

                # Get results
                response_result = requests.get(
                    f'https://api.cloud.llamaindex.ai/api/v1/parsing/job/{job_id}/result/json',
                    headers=headers
                )
                response_result.raise_for_status()
                st.success("Parsing completed successfully!")
                json_string = response_result.json()["pages"][0]["items"][0]['value'].replace('```json\n', '').replace('\n```', '')
                st.json(json.loads(json_string))

            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred: {str(e)}")