from pipeline_rag.pdf_loader import extract_text
from pipeline_rag.chunker import split_text
from pipeline_rag.embeddings import build_index
from pipeline_rag.retriever import retrieve
from pipeline_rag.generator import answer_question


class PDFChatbot:

    def __init__(self, api_key):
        self.api_key = api_key
        self.chunks = None
        self.index = None
        self.history = []

    def load_pdf(self, pdf_path):

        text = extract_text(pdf_path)

        self.chunks = split_text(text)

        self.index, _ = build_index(
            self.chunks
        )

    def ask(self, question):

        context = retrieve(
            question,
            self.chunks,
            self.index
        )

        answer = answer_question(
            question,
            context,
            self.api_key
        )

        self.history.append({
            "question": question,
            "answer": answer,
        })

        return answer