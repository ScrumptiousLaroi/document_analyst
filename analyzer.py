from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import os
import re
from datetime import datetime
from typing import List, Dict, Any

# Import the PDF processing function from our other file
from pdf_processor import extract_sections_from_pdf

def check_model_availability(model_name: str = 'all-MiniLM-L6-v2') -> bool:
    """
    Check if the model is available locally in the project directory.
    
    Args:
        model_name: Name of the sentence transformer model
        
    Returns:
        bool: True if model is available locally, False otherwise
    """
    try:
        # Get the local model path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "models")
        local_model_path = os.path.join(models_dir, model_name)
        
        # Check if the model directory exists and has required files
        if not os.path.exists(local_model_path):
            return False
            
        # Check for essential model files
        required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
        model_files = os.listdir(local_model_path)
        
        # Check if at least some model files exist (different models may have different file structures)
        has_config = any('config' in f for f in model_files)
        has_model = any('pytorch_model' in f or 'model' in f for f in model_files)
        
        return has_config and has_model
        
    except Exception:
        return False

def extract_section_title_from_content(content: str, page_num: int) -> str:
    """
    Extract a meaningful section title from content when TOC is not available.
    
    Args:
        content (str): The text content of the section
        page_num (int): The page number
        
    Returns:
        str: A meaningful section title
    """
    if not content.strip():
        return f"Page {page_num}"
    
    # Split content into lines and clean them
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    if not lines:
        return f"Page {page_num}"
    
    # Look for potential titles in the first few lines
    for line in lines[:5]:  # Check first 5 lines
        # Remove common non-title patterns
        cleaned_line = re.sub(r'^\d+[\.\s]*', '', line)  # Remove leading numbers
        cleaned_line = re.sub(r'^Page\s+\d+', '', cleaned_line, flags=re.IGNORECASE)  # Remove "Page X"
        cleaned_line = cleaned_line.strip()
        
        # Check if line looks like a title
        if (len(cleaned_line) > 5 and len(cleaned_line) < 100 and 
            not cleaned_line.endswith('.') and 
            not cleaned_line.startswith('http') and
            not re.search(r'\d{4}-\d{2}-\d{2}', cleaned_line) and  # No dates
            cleaned_line.count(' ') < 15):  # Not too many words
            
            # Prioritize lines that are all caps or title case
            if cleaned_line.isupper() or cleaned_line.istitle():
                return cleaned_line
            elif any(word[0].isupper() for word in cleaned_line.split() if word):
                return cleaned_line
    
    # Fallback: use first meaningful sentence
    sentences = re.split(r'[.!?]+', content)
    for sentence in sentences[:3]:
        sentence = sentence.strip()
        if len(sentence) > 10 and len(sentence) < 150:
            return sentence + "..."
    
    return f"Page {page_num}"

def calculate_accuracy_score(sections: List[Dict], query_embedding, model) -> float:
    """
    Calculate an accuracy score based on semantic relevance and content quality.
    
    Args:
        sections: List of extracted sections
        query_embedding: The query embedding for comparison
        model: The sentence transformer model
        
    Returns:
        float: Accuracy score between 0 and 1
    """
    if not sections:
        return 0.0
    
    # Calculate semantic relevance scores
    section_contents = [sec["content"] for sec in sections if sec["content"].strip()]
    if not section_contents:
        return 0.0
        
    section_embeddings = model.encode(section_contents, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, section_embeddings)
    
    # Metrics for accuracy calculation
    avg_relevance = float(torch.mean(cosine_scores).item())
    max_relevance = float(torch.max(cosine_scores).item())
    
    # Content quality metrics
    avg_content_length = np.mean([len(content) for content in section_contents])
    sections_with_meaningful_titles = sum(1 for sec in sections 
                                        if not sec["section_title"].startswith("Page "))
    title_quality_ratio = sections_with_meaningful_titles / len(sections)
    
    # Combine metrics for final accuracy score
    # Weighted combination: relevance (60%), content length (20%), title quality (20%)
    content_length_score = min(avg_content_length / 1000, 1.0)  # Normalize to 0-1
    
    accuracy = (avg_relevance * 0.4 + 
               max_relevance * 0.2 + 
               content_length_score * 0.2 + 
               title_quality_ratio * 0.2)
    
    return min(accuracy, 1.0)  # Cap at 1.0

def run_analysis(input_data):
    """
    Main function to run the entire analysis pipeline based on the project plan.
    """
    # --- Phase 1: Initialization and Setup ---
    print("Phase 1: Initializing and loading model...")
    
    model_name = 'all-MiniLM-L6-v2'
    
    # Check if model is available locally first
    if not check_model_availability(model_name):
        print(f"⚠️  Model '{model_name}' not found locally.")
        print("Please run 'python download_model.py' first with internet connection.")
        print("This will download and cache the model in the project directory for offline usage.")
        return {}
    
    try:
        # Load the model from local directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "models")
        local_model_path = os.path.join(models_dir, model_name)
        
        model = SentenceTransformer(local_model_path)
        print(f"✅ Model '{model_name}' loaded from local directory: {local_model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please run 'python download_model.py' to re-download the model.")
        return {}

    persona = input_data["persona"]["role"]
    job_to_be_done = input_data["job_to_be_done"]["task"]
    document_infos = input_data["documents"]
    
    # Generate processing timestamp
    processing_timestamp = datetime.now().isoformat()
    
    # --- Phase 2: PDF Processing and Section Aggregation ---
    print("Phase 2: Processing PDFs and extracting sections...")
    all_sections = []
    for doc_info in document_infos:
        # Assuming documents are in a 'documents' subdirectory
        doc_path = os.path.join("documents", doc_info["filename"])
        if os.path.exists(doc_path):
            sections = extract_sections_from_pdf(doc_path)
            # Improve section titles for pages without TOC
            for section in sections:
                if section["section_title"].startswith("Page "):
                    improved_title = extract_section_title_from_content(
                        section["content"], section["page_number"]
                    )
                    if not improved_title.startswith("Page "):
                        section["section_title"] = improved_title
            all_sections.extend(sections)
        else:
            print(f"Warning: Document not found at {doc_path}")

    if not all_sections:
        print("Error: No sections were extracted. Cannot proceed.")
        return {}

    # --- Phase 3: Semantic Analysis and Prioritization ---
    print("Phase 3: Performing semantic analysis and ranking...")
    # Create a more comprehensive query
    query = f"As a {persona}, I need to {job_to_be_done}. What information is most relevant for planning and execution?"
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Filter out sections with minimal content
    meaningful_sections = [sec for sec in all_sections if len(sec["content"].strip()) > 50]
    
    if not meaningful_sections:
        meaningful_sections = all_sections  # Fallback to all sections
    
    section_contents = [sec["content"] for sec in meaningful_sections]
    section_embeddings = model.encode(section_contents, convert_to_tensor=True)
    
    cosine_scores = util.cos_sim(query_embedding, section_embeddings)

    for i, section in enumerate(meaningful_sections):
        section["relevance_score"] = cosine_scores[0][i].item()

    ranked_sections = sorted(meaningful_sections, key=lambda x: x["relevance_score"], reverse=True)
    
    for i, section in enumerate(ranked_sections):
        section["importance_rank"] = i + 1

    # Calculate accuracy score
    accuracy_score = calculate_accuracy_score(ranked_sections, query_embedding, model)

    # --- Phase 4: Content Refinement and Snippet Extraction ---
    print("Phase 4: Refining top sections for key snippets...")
    # Take only top 5 sections for analysis to keep output concise
    top_n = 5
    top_sections = ranked_sections[:top_n]
    
    subsection_analysis_results = []
    
    for section in top_sections:
        # Better sentence splitting that preserves context
        sentences = re.split(r'(?<=[.!?])\s+', section["content"])
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]

        if not sentences:
            # If no good sentences, use the whole content if it's short enough
            if len(section["content"].strip()) < 500:
                refined_text = section["content"].strip()
            else:
                refined_text = section["content"].strip()[:400] + "..."
        else:
            # Find the most relevant sentences (top 3)
            sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
            sentence_scores = util.cos_sim(query_embedding, sentence_embeddings)
            
            # Get top 3 sentences and combine them
            top_sentence_indices = torch.topk(sentence_scores[0], k=min(3, len(sentences))).indices
            top_sentences = [sentences[i] for i in sorted(top_sentence_indices.cpu().numpy())]
            refined_text = " ".join(top_sentences)

        subsection_analysis_results.append({
            "document": section["document"],
            "refined_text": refined_text,
            "page_number": section["page_number"]
        })

    # --- Phase 5: Final Output Construction ---
    print("Phase 5: Constructing final JSON output...")
    print(f"Calculated accuracy score: {accuracy_score:.3f}")
    
    final_output = {
        "metadata": {
            "input_documents": [doc["filename"] for doc in document_infos],
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": processing_timestamp
        },
        "extracted_sections": [
            {
                "document": sec["document"],
                "section_title": sec["section_title"],
                "importance_rank": sec["importance_rank"],
                "page_number": sec["page_number"]
            } for sec in ranked_sections[:5]  # Limit to top 5 sections only
        ],
        "subsection_analysis": subsection_analysis_results
    }
    
    return final_output
