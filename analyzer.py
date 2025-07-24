#!/usr/bin/env python3
"""
Optimized PDF Analyzer - Ensures High Accuracy by Fixing Core Issues
This version addresses the document diversity and content selection problems
"""

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
    """Check if the model is available locally in the project directory."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "models")
        local_model_path = os.path.join(models_dir, model_name)
        
        if not os.path.exists(local_model_path):
            return False
            
        model_files = os.listdir(local_model_path)
        has_config = any('config' in f for f in model_files)
        has_model = any('pytorch_model' in f or 'model' in f for f in model_files)
        
        return has_config and has_model
        
    except Exception:
        return False

def extract_section_title_from_content(content: str, page_num: int) -> str:
    """Extract a meaningful section title from content when TOC is not available."""
    if not content.strip():
        return f"Page {page_num}"
    
    lines = [line.strip() for line in content.split('\\n') if line.strip()]
    
    if not lines:
        return f"Page {page_num}"
    
    # Look for potential titles in the first few lines
    for line in lines[:5]:
        cleaned_line = re.sub(r'^\\d+[\\.\\s]*', '', line)
        cleaned_line = re.sub(r'^Page\\s+\\d+', '', cleaned_line, flags=re.IGNORECASE)
        cleaned_line = cleaned_line.strip()
        
        if (len(cleaned_line) > 5 and len(cleaned_line) < 100 and 
            not cleaned_line.endswith('.') and 
            not cleaned_line.startswith('http') and
            not re.search(r'\\d{4}-\\d{2}-\\d{2}', cleaned_line) and
            cleaned_line.count(' ') < 15):
            
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

def ensure_document_diversity(sections: List[Dict], min_docs: int = 4) -> List[Dict]:
    """Ensure we have sections from multiple documents for diversity."""
    # Group sections by document
    docs_sections = {}
    for section in sections:
        doc = section.get("document", "unknown")
        if doc not in docs_sections:
            docs_sections[doc] = []
        docs_sections[doc].append(section)
    
    # Select best sections from each document
    diverse_sections = []
    
    # First, get at least one section from each document
    for doc, doc_sections in docs_sections.items():
        # Sort by relevance score and take the best one
        best_section = max(doc_sections, key=lambda x: x.get("relevance_score", 0))
        diverse_sections.append(best_section)
    
    # If we need more sections, add the next best from any document
    remaining_sections = [s for s in sections if s not in diverse_sections]
    remaining_sections.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    # Add more sections until we reach the desired count
    target_count = max(min_docs, 5)  # At least 5 sections total
    for section in remaining_sections:
        if len(diverse_sections) >= target_count:
            break
        diverse_sections.append(section)
    
    # Re-rank the diverse sections
    diverse_sections.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    # Update importance ranks
    for i, section in enumerate(diverse_sections):
        section["importance_rank"] = i + 1
    
    return diverse_sections

def run_analysis(input_data):
    """Main function to run the entire analysis pipeline with optimizations."""
    print("Phase 1: Initializing and loading model...")
    
    model_name = 'all-MiniLM-L6-v2'
    
    if not check_model_availability(model_name):
        print(f"⚠️  Model '{model_name}' not found locally.")
        print("Please run 'python download_model.py' first with internet connection.")
        return {}
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "models")
        local_model_path = os.path.join(models_dir, model_name)
        
        model = SentenceTransformer(local_model_path)
        print(f"✅ Model '{model_name}' loaded from local directory: {local_model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return {}

    persona = input_data["persona"]["role"]
    job_to_be_done = input_data["job_to_be_done"]["task"]
    document_infos = input_data["documents"]
    
    processing_timestamp = datetime.now().isoformat()
    
    print("Phase 2: Processing PDFs and extracting sections...")
    all_sections = []
    
    for doc_info in document_infos:
        doc_path = os.path.join("documents", doc_info["filename"])
        if os.path.exists(doc_path):
            sections = extract_sections_from_pdf(doc_path)
            # Improve section titles
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

    print("Phase 3: Performing semantic analysis and ranking...")
    
    # Create enhanced query for better matching
    base_query = f"As a {persona}, I need to {job_to_be_done}"
    
    # Add context-specific keywords based on persona
    if "travel" in persona.lower() or "plan" in job_to_be_done.lower():
        enhanced_query = f"{base_query}. Focus on destinations, activities, practical tips, accommodations, and experiences."
    else:
        enhanced_query = f"{base_query}. What information is most relevant for planning and execution?"
    
    query_embedding = model.encode(enhanced_query, convert_to_tensor=True)
    
    # Filter meaningful sections
    meaningful_sections = [sec for sec in all_sections if len(sec["content"].strip()) > 30]
    
    if not meaningful_sections:
        meaningful_sections = all_sections
    
    section_contents = [sec["content"] for sec in meaningful_sections]
    section_embeddings = model.encode(section_contents, convert_to_tensor=True)
    
    cosine_scores = util.cos_sim(query_embedding, section_embeddings)

    for i, section in enumerate(meaningful_sections):
        section["relevance_score"] = cosine_scores[0][i].item()

    # Sort by relevance
    ranked_sections = sorted(meaningful_sections, key=lambda x: x["relevance_score"], reverse=True)
    
    # CRITICAL: Ensure document diversity
    diverse_sections = ensure_document_diversity(ranked_sections, min_docs=4)

    print("Phase 4: Refining top sections for key snippets...")
    
    # Take top sections for subsection analysis
    top_sections = diverse_sections[:5]
    subsection_analysis_results = []
    
    for section in top_sections:
        sentences = re.split(r'(?<=[.!?])\\s+', section["content"])
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]

        if not sentences:
            if len(section["content"].strip()) < 500:
                refined_text = section["content"].strip()
            else:
                refined_text = section["content"].strip()[:400] + "..."
        else:
            # Find most relevant sentences
            sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
            sentence_scores = util.cos_sim(query_embedding, sentence_embeddings)
            
            # Get top 4 sentences for richer content
            top_sentence_indices = torch.topk(sentence_scores[0], k=min(4, len(sentences))).indices
            top_sentences = [sentences[i] for i in sorted(top_sentence_indices.cpu().numpy())]
            refined_text = " ".join(top_sentences)

        subsection_analysis_results.append({
            "document": section["document"],
            "refined_text": refined_text,
            "page_number": section["page_number"]
        })

    print("Phase 5: Constructing final JSON output...")
    
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
            } for sec in diverse_sections[:5]  # Top 5 diverse sections
        ],
        "subsection_analysis": subsection_analysis_results
    }
    
    return final_output
