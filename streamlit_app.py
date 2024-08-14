import streamlit as st
import requests
import time
import json
from PIL import Image
import io
import fitz  # PyMuPDF

# API configuration
API_URL = 'https://api.cloud.llamaindex.ai/api/v1/parsing/upload'
API_KEY = 'llx-kHVYZKk7lHFXXL7QQ4S4ZPdvFqs1MypudZUYF4Jv6Jds7YTx'  # Consider using st.secrets for API keys

headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {API_KEY}',
}


def pdf_to_images(pdf_bytes):
    images = []
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in pdf_document:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images


# Streamlit UI
st.title("📄 Document Parsing")
st.write("Upload a document below for parsing!")

uploaded_file = st.file_uploader("Upload a document (.pdf, .png, .jpg)", type=["pdf", 'png', 'jpg'])

if uploaded_file:
    st.write("File uploaded successfully!")

    # Prepare the file for API request
    file_bytes = uploaded_file.getvalue()

    parsing_instruction = """
    File is an invoice. Parse the following fields:
    - Invoice number
    - Date
    - Sender name
    - Sender address
    - Receiver name
    - Receiver address
    - VAT number
    - Total sum

    Validate the following fields against the provided values:
    - Receiver name: "UAB Linėja transport"
    - Receiver address: "Didžioji g. 38, Kėdainiai, LT-57257, Lithuania"
    - VAT number: "LT100006144519"

    For each validated field:
    1. If the parsed value is similar or matches the provided value:
       - Replace the parsed value with the provided value
       - Add a field "[fieldname]_validated" with value "success"
    2. If the parsed value is significantly different:
       - Keep the original parsed value
       - Add a field "[fieldname]_validated" with value "failed"

    Return a JSON object with all parsed fields and their validation statuses.
    """

    files = {
        'language': (None, 'lt'),
        'parsing_instruction': (None, parsing_instruction),
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
                result_pages = response_result.json()["pages"]

                # Convert PDF to images if it's a PDF file
                if uploaded_file.type == "application/pdf":
                    pdf_images = pdf_to_images(file_bytes)

                for i, page in enumerate(result_pages):
                    st.subheader(f"Page {i + 1}")

                    # Create two columns
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Document Image")
                        # Display the image
                        if uploaded_file.type == "application/pdf":
                            st.image(pdf_images[i], caption=f"Page {i + 1}", use_column_width=True)
                        else:
                            image = Image.open(io.BytesIO(file_bytes))
                            st.image(image, caption=f"Page {i + 1}", use_column_width=True)

                    with col2:
                        st.subheader("Parsed Data")
                        # Display parsed JSON
                        json_string = page["items"][0]['value'].replace('```json\n', '').replace('\n```', '')
                        parsed_data = json.loads(json_string)
                        for key, value in parsed_data.items():
                            st.write(f"**{key}:** {value}")

                    st.markdown("---")  # Add a horizontal line for separation

            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred: {str(e)}")