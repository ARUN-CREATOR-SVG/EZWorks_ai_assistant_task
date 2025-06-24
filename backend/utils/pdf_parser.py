import fitz

def parse_document(file_path):
    if file_path.endswith(".pdf"):
        with fitz.open(file_path) as doc:
            paragraphs = []
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                for para_num, para in enumerate(text.split("\n\n"), start=1):
                    if para.strip():
                        paragraphs.append({
                            "content": para.strip(),
                            "page": page_num,
                            "paragraph": para_num
                        })
            return paragraphs
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n\n")
            return [{"content": para.strip(), "page": 1, "paragraph": i + 1} for i, para in enumerate(lines) if para.strip()]
    else:
        return []
