import pymupdf


def load_pdf(pdf_path: str) -> str:
    doc = pymupdf.open(pdf_path)
    text = ""
    for page in doc:
        page_text = page.get_text()
        if page_text:
            text += page_text + "\n"

    doc.close()
    if not text.strip():
        raise ValueError("No text found in PDF. Is it a scanned document? Check and then Try Again.")

    return text