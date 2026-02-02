import io
from fastapi import APIRouter, File, UploadFile
from typing import Any
from pdfminer.high_level import extract_text

router = APIRouter()

@router.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)) -> Any:
    """
    Upload a PDF file.
    """
    print(f"Received PDF file: {file.filename}")    
    pdf_bytes = await file.read()
    text = extract_text(io.BytesIO(pdf_bytes))
    print(text[:100])

    return {"filename": file.filename, "content_type": file.content_type, "text": text}



@router.post("/upload/image")
async def upload_image(file: UploadFile = File(...)) -> Any:
    """
    Upload an image file.
    """
    print(f"Received image file: {file.filename}")
    return {"filename": file.filename, "content_type": file.content_type, "message": "Image received"} 