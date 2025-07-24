#!/usr/bin/env python3
"""
Debug Test Runner - Single Collection Analysis
"""

import json
import os
import sys
from pathlib import Path
import shutil

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fixed_enhanced_analyzer import run_analysis

def test_single_collection():
    """Test Collection 1 and show detailed output"""
    collection_path = Path("test_pdf/Collection 1")
    
    # Load input data
    with open(collection_path / "challenge1b_input.json", 'r') as f:
        input_data = json.load(f)
    
    # Copy PDFs to documents folder
    documents_dir = Path("documents")
    documents_dir.mkdir(exist_ok=True)
    
    pdf_folder = collection_path / "PDF"
    pdf_files = list(pdf_folder.glob("*.pdf"))
    
    for pdf_file in pdf_files:
        target_path = documents_dir / pdf_file.name
        shutil.copy2(pdf_file, target_path)
    
    # Update input data
    input_data["documents"] = [{"filename": pdf.name, "type": "pdf"} for pdf in pdf_files]
    
    print("Input data:")
    print(json.dumps(input_data, indent=2))
    print("\\n" + "="*50)
    
    # Run analysis
    result = run_analysis(input_data)
    
    print("\\nActual output:")
    print(json.dumps(result, indent=2))
    
    # Load expected output
    with open(collection_path / "challenge1b_output.json", 'r') as f:
        expected = json.load(f)
    
    print("\\n" + "="*50)
    print("\\nExpected output:")
    print(json.dumps(expected, indent=2))
    
    # Cleanup
    for pdf_file in pdf_files:
        target_path = documents_dir / pdf_file.name
        if target_path.exists():
            target_path.unlink()

if __name__ == "__main__":
    test_single_collection()
