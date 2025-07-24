import fitz  # PyMuPDF
import os
import re

def clean_extracted_text(text):
    """
    Clean extracted text by removing excessive whitespace and fixing common OCR issues.
    
    Args:
        text (str): Raw extracted text
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common line break issues
    text = re.sub(r'([a-z])\s*\n\s*([a-z])', r'\1 \2', text)
    
    # Remove page numbers and headers/footers patterns
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'^Page \d+.*?\n', '', text, flags=re.MULTILINE)
    
    return text.strip()

def extract_title_from_page_content(page_text):
    """
    Extract a potential title from page content by analyzing formatting and position.
    
    Args:
        page_text (str): Text content of the page
        
    Returns:
        str: Extracted title or None if no clear title found
    """
    if not page_text.strip():
        return None
    
    lines = [line.strip() for line in page_text.split('\n') if line.strip()]
    if not lines:
        return None
    
    # Look for title patterns in first few lines
    for i, line in enumerate(lines[:5]):
        # Skip very short lines or lines with numbers only
        if len(line) < 3 or line.isdigit():
            continue
            
        # Check if line looks like a title
        if (5 <= len(line) <= 100 and 
            not line.endswith('.') and
            not line.startswith('http') and
            line.count(' ') <= 10):
            
            # Prioritize lines that are formatted like titles
            if (line.isupper() or 
                line.istitle() or 
                all(word[0].isupper() for word in line.split() if word and word.isalpha())):
                return line
    
    return None

def extract_sections_from_pdf(doc_path):
    """
    Extracts text sections from a PDF. Uses the Table of Contents if available,
    otherwise treats each page as a section.

    Args:
        doc_path (str): The file path to the PDF.

    Returns:
        list: A list of dictionaries, where each dict represents a section.
    """
    try:
        doc = fitz.open(doc_path)
    except Exception as e:
        print(f"Error opening {doc_path}: {e}")
        return []

    sections = []
    toc = doc.get_toc()
    doc_filename = os.path.basename(doc_path)

    if toc:
        # Method 1: Use Table of Contents to define sections
        for i, item in enumerate(toc):
            level, title, page_num = item
            start_page = page_num - 1
            
            # Determine the end page for the section
            if i + 1 < len(toc):
                # The end page is the page number of the *next* ToC item
                end_page = toc[i+1][2] - 1
            else:
                # If it's the last ToC item, it goes to the end of the document
                end_page = len(doc)
            
            content = ""
            # Extract text from the page range
            for page_index in range(start_page, end_page):
                if page_index < len(doc):
                    page_text = doc[page_index].get_text()
                    content += clean_extracted_text(page_text) + "\n"

            sections.append({
                "document": doc_filename,
                "section_title": title.strip(),
                "content": content.strip(),
                "page_number": start_page + 1
            })
    else:
        # Fallback Method: Treat each page as a section if no ToC is found
        # But try to extract meaningful titles from content
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            cleaned_content = clean_extracted_text(page_text)
            
            # Try to extract a meaningful title from the page content
            extracted_title = extract_title_from_page_content(page_text)
            
            if extracted_title:
                section_title = extracted_title
            else:
                section_title = f"Page {page_num + 1}"
            
            sections.append({
                "document": doc_filename,
                "section_title": section_title,
                "content": cleaned_content,
                "page_number": page_num + 1
            })
    
    doc.close()
    return sections
