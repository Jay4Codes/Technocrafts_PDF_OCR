import streamlit as st
import fitz  # PyMuPDF
import tempfile
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR

# Set page configuration
st.set_page_config(
    page_title="Technocrafts Solar Panel Drawing OCR",
    page_icon="☀️",
    layout="wide"
)

# Initialize PaddleOCR
@st.cache_resource
def load_ocr_model():
    return PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

ocr = load_ocr_model()

def extract_text_from_pdf(pdf_file):
    """Extract text from each page of a PDF using PaddleOCR"""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name
    
    # Open the PDF with PyMuPDF
    doc = fitz.open(tmp_path)
    page_texts = []
    
    # Process each page
    for page_num in range(len(doc)):
        # Get the page
        page = doc.load_page(page_num)
        
        # Convert to image
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        img_bytes = pix.tobytes("png")
        
        # Convert bytes to numpy array for OpenCV
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run OCR on the image
        result = ocr.ocr(img, cls=True)
        
        # Extract and combine text
        page_text = ""
        if result[0]:  # Check if results exist
            for line in result[0]:
                if line[1][0]:  # Check if text was detected
                    page_text += line[1][0] + "\n"
        
        page_texts.append(page_text)
    
    # Close the document and delete the temporary file
    doc.close()
    os.unlink(tmp_path)
    
    return page_texts

# App title and description
st.title("Technocrafts Solar Panel Drawing OCR")
st.markdown("""
Upload engineering drawings of solar panels to extract text information using OCR.
This application uses PaddleOCR to recognize text from PDF documents.
""")

# File uploader
pdf_file = st.file_uploader("Upload a Solar Panel Engineering Drawing PDF", type=["pdf"])

if pdf_file is not None:
    with st.spinner("Processing PDF with OCR... This may take a minute."):
        # Extract text from PDF
        page_texts = extract_text_from_pdf(pdf_file)
        
        # Display success message
        st.success(f"Successfully processed {len(page_texts)} pages from the PDF!")
        
        # Create a dropdown to select pages
        st.subheader("Select a page to view extracted text")
        page_num = st.selectbox(
            "Page Number", 
            options=list(range(1, len(page_texts) + 1)),
            format_func=lambda x: f"Page {x}"
        )
        
        # Display the selected page's text
        st.subheader(f"Extracted Text from Page {page_num}")
        
        # Show the extracted text in a text area
        selected_text = page_texts[page_num - 1]
        if selected_text.strip():
            st.text_area("Extracted Text", value=selected_text, height=400)
        else:
            st.warning("No text was detected on this page or the OCR failed to recognize text.")
        
        # Add download button for all extracted text
        all_text = "\n\n--- PAGE BREAK ---\n\n".join([f"PAGE {i+1}:\n{text}" for i, text in enumerate(page_texts)])
        st.download_button(
            label="Download All Extracted Text",
            data=all_text,
            file_name="technocrafts_solar_panel_ocr.txt",
            mime="text/plain"
        )
else:
    st.info("Please upload a PDF file to begin.")

# Add information footer
st.markdown("---")
st.markdown("""
**About This Tool**
- This tool is designed for Technocrafts solar panel engineering drawings
- Powered by PaddleOCR for text recognition
- Created by Technocrafts Engineering Team
""")