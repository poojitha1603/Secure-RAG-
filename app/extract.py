"""Turn an uploaded file into plain text."""
import io


class UnsupportedFile(ValueError):
    pass


def extract_text(filename: str, data: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="replace").strip()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()
        except Exception as e:
            raise UnsupportedFile(f"could not read PDF: {e}") from e
    raise UnsupportedFile("only .txt, .md and .pdf files are supported")