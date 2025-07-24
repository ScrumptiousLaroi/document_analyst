#!/usr/bin/env python3
"""
Simple Test Runner for PDF Analyzer
Tests the analyzer against expected outputs and calculates accuracy
"""

import json
import os
import sys
from pathlib import Path
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Any
import difflib

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fixed_enhanced_analyzer import run_analysis

class SimpleTestRunner:
    def __init__(self, test_directory: str = "test_pdf"):
        self.test_directory = Path(test_directory)
        self.results = []
        
    def load_json_file(self, file_path: Path) -> Dict:
        """Load a JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return {}
    
    def copy_pdfs_to_documents(self, pdf_folder: Path) -> List[str]:
        """Copy PDFs to documents folder and return filenames"""
        import shutil
        
        documents_dir = Path("documents")
        documents_dir.mkdir(exist_ok=True)
        
        pdf_files = list(pdf_folder.glob("*.pdf"))
        copied_files = []
        
        for pdf_file in pdf_files:
            target_path = documents_dir / pdf_file.name
            try:
                shutil.copy2(pdf_file, target_path)
                copied_files.append(pdf_file.name)
                print(f"Copied {pdf_file.name} to documents/")
            except Exception as e:
                print(f"Error copying {pdf_file.name}: {e}")
        
        return copied_files
    
    def cleanup_documents(self, filenames: List[str]):
        """Remove copied PDF files from documents folder"""
        documents_dir = Path("documents")
        for filename in filenames:
            file_path = documents_dir / filename
            try:
                if file_path.exists():
                    file_path.unlink()
                    print(f"Cleaned up {filename}")
            except Exception as e:
                print(f"Error cleaning up {filename}: {e}")
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using difflib"""
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts
        text1 = ' '.join(text1.lower().split())
        text2 = ' '.join(text2.lower().split())
        
        # Calculate similarity using SequenceMatcher
        similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
        return similarity
    
    def calculate_structural_similarity(self, expected: Dict, actual: Dict) -> float:
        """Calculate structural similarity between expected and actual results"""
        if not expected or not actual:
            return 0.0
        
        # Check if both have the required structure
        expected_keys = set(expected.keys())
        actual_keys = set(actual.keys())
        
        # Calculate key overlap
        key_overlap = len(expected_keys & actual_keys) / max(len(expected_keys), 1)
        
        # Check specific structures
        structure_score = 0.0
        checks = 0
        
        # Check metadata
        if 'metadata' in expected and 'metadata' in actual:
            exp_meta_keys = set(expected['metadata'].keys())
            act_meta_keys = set(actual['metadata'].keys())
            meta_overlap = len(exp_meta_keys & act_meta_keys) / max(len(exp_meta_keys), 1)
            structure_score += meta_overlap
            checks += 1
        
        # Check extracted_sections
        if 'extracted_sections' in expected and 'extracted_sections' in actual:
            exp_sections = expected['extracted_sections']
            act_sections = actual['extracted_sections']
            
            if isinstance(exp_sections, list) and isinstance(act_sections, list):
                # Compare number of sections (normalized)
                section_count_similarity = min(len(act_sections), len(exp_sections)) / max(len(exp_sections), 1)
                structure_score += section_count_similarity
                checks += 1
        
        # Check subsection_analysis
        if 'subsection_analysis' in expected and 'subsection_analysis' in actual:
            exp_subsections = expected['subsection_analysis']
            act_subsections = actual['subsection_analysis']
            
            if isinstance(exp_subsections, list) and isinstance(act_subsections, list):
                subsection_count_similarity = min(len(act_subsections), len(exp_subsections)) / max(len(exp_subsections), 1)
                structure_score += subsection_count_similarity
                checks += 1
        
        if checks > 0:
            structure_score /= checks
        
        # Combined score: 50% key overlap, 50% structure details
        return (key_overlap + structure_score) / 2
    
    def calculate_content_similarity(self, expected: Dict, actual: Dict) -> float:
        """Calculate content similarity between expected and actual results"""
        if not expected or not actual:
            return 0.0
        
        content_similarities = []
        
        # Compare extracted sections content
        if 'extracted_sections' in expected and 'extracted_sections' in actual:
            exp_sections = expected['extracted_sections']
            act_sections = actual['extracted_sections']
            
            if isinstance(exp_sections, list) and isinstance(act_sections, list):
                # Extract section titles for comparison
                exp_titles = [section.get('section_title', '') for section in exp_sections]
                act_titles = [section.get('section_title', '') for section in act_sections]
                
                exp_titles_text = ' '.join(exp_titles)
                act_titles_text = ' '.join(act_titles)
                
                if exp_titles_text and act_titles_text:
                    title_similarity = self.calculate_text_similarity(exp_titles_text, act_titles_text)
                    content_similarities.append(title_similarity)
        
        # Compare subsection analysis content
        if 'subsection_analysis' in expected and 'subsection_analysis' in actual:
            exp_subsections = expected['subsection_analysis']
            act_subsections = actual['subsection_analysis']
            
            if isinstance(exp_subsections, list) and isinstance(act_subsections, list):
                # Extract refined text for comparison
                exp_texts = [sub.get('refined_text', '') for sub in exp_subsections]
                act_texts = [sub.get('refined_text', '') for sub in act_subsections]
                
                exp_combined_text = ' '.join(exp_texts)
                act_combined_text = ' '.join(act_texts)
                
                if exp_combined_text and act_combined_text:
                    text_similarity = self.calculate_text_similarity(exp_combined_text, act_combined_text)
                    content_similarities.append(text_similarity)
        
        # Return average similarity
        return sum(content_similarities) / len(content_similarities) if content_similarities else 0.0
    
    def calculate_overall_accuracy(self, expected: Dict, actual: Dict) -> Tuple[float, Dict]:
        """Calculate overall accuracy with detailed breakdown"""
        if not expected or not actual:
            return 0.0, {"error": "Missing expected or actual data"}
        
        # Calculate different aspects of accuracy
        structural_similarity = self.calculate_structural_similarity(expected, actual)
        content_similarity = self.calculate_content_similarity(expected, actual)
        
        # Overall accuracy: weighted average
        overall_accuracy = (structural_similarity * 0.4 + content_similarity * 0.6) * 100
        
        breakdown = {
            "structural_similarity": structural_similarity * 100,
            "content_similarity": content_similarity * 100,
            "overall_accuracy": overall_accuracy
        }
        
        return overall_accuracy, breakdown
    
    def test_collection(self, collection_name: str) -> Dict:
        """Test a single collection"""
        print(f"\n{'='*50}")
        print(f"Testing Collection: {collection_name}")
        print(f"{'='*50}")
        
        collection_path = self.test_directory / collection_name
        if not collection_path.exists():
            return {"error": f"Collection not found: {collection_path}"}
        
        try:
            # Load input and expected output
            input_files = list(collection_path.glob("*input*.json"))
            output_files = list(collection_path.glob("*output*.json"))
            
            if not input_files:
                return {"error": f"No input JSON found in {collection_name}"}
            if not output_files:
                return {"error": f"No output JSON found in {collection_name}"}
            
            input_data = self.load_json_file(input_files[0])
            expected_output = self.load_json_file(output_files[0])
            
            if not input_data:
                return {"error": f"Failed to load input data for {collection_name}"}
            if not expected_output:
                return {"error": f"Failed to load expected output for {collection_name}"}
            
            # Copy PDFs to documents folder
            pdf_folder = collection_path / "PDF"
            if not pdf_folder.exists():
                return {"error": f"No PDF folder found in {collection_name}"}
            
            copied_files = self.copy_pdfs_to_documents(pdf_folder)
            if not copied_files:
                return {"error": f"No PDFs found or copied in {collection_name}"}
            
            # Update input data with copied filenames
            input_data["documents"] = [{"filename": filename, "type": "pdf"} for filename in copied_files]
            
            print(f"Running analyzer on {len(copied_files)} PDFs...")
            
            # Run the analyzer
            actual_output = run_analysis(input_data)
            
            # Clean up copied files
            self.cleanup_documents(copied_files)
            
            if not actual_output:
                return {"error": f"Analyzer returned empty result for {collection_name}"}
            
            # Calculate accuracy
            accuracy, breakdown = self.calculate_overall_accuracy(expected_output, actual_output)
            
            result = {
                "collection": collection_name,
                "accuracy": accuracy,
                "breakdown": breakdown,
                "expected_structure": list(expected_output.keys()) if expected_output else [],
                "actual_structure": list(actual_output.keys()) if actual_output else [],
                "pdf_files_processed": copied_files,
                "timestamp": datetime.now().isoformat()
            }
            
            print(f"Accuracy: {accuracy:.2f}%")
            print(f"Breakdown: {breakdown}")
            
            return result
            
        except Exception as e:
            print(f"Error testing {collection_name}: {e}")
            traceback.print_exc()
            return {"error": str(e), "collection": collection_name}
    
    def run_all_tests(self) -> List[Dict]:
        """Run tests on all collections"""
        print("PDF Analyzer Comprehensive Testing")
        print("=" * 50)
        
        if not self.test_directory.exists():
            print(f"Test directory not found: {self.test_directory}")
            return []
        
        # Find all collection directories
        collections = [d.name for d in self.test_directory.iterdir() 
                      if d.is_dir() and d.name.startswith('Collection')]
        collections.sort()
        
        if not collections:
            print("No collection directories found")
            return []
        
        print(f"Found {len(collections)} collections: {', '.join(collections)}")
        
        results = []
        for collection in collections:
            result = self.test_collection(collection)
            results.append(result)
        
        return results
    
    def generate_summary_report(self, results: List[Dict]) -> Dict:
        """Generate a summary report of all test results"""
        if not results:
            return {"error": "No test results to summarize"}
        
        # Filter out error results
        valid_results = [r for r in results if "error" not in r and "accuracy" in r]
        error_results = [r for r in results if "error" in r]
        
        if not valid_results:
            return {
                "total_collections": len(results),
                "successful_tests": 0,
                "failed_tests": len(error_results),
                "errors": error_results
            }
        
        # Calculate overall statistics
        accuracies = [r["accuracy"] for r in valid_results]
        avg_accuracy = sum(accuracies) / len(accuracies)
        min_accuracy = min(accuracies)
        max_accuracy = max(accuracies)
        
        # Determine if improvement is needed
        needs_improvement = avg_accuracy < 90.0
        
        summary = {
            "test_timestamp": datetime.now().isoformat(),
            "total_collections": len(results),
            "successful_tests": len(valid_results),
            "failed_tests": len(error_results),
            "average_accuracy": avg_accuracy,
            "min_accuracy": min_accuracy,
            "max_accuracy": max_accuracy,
            "needs_improvement": needs_improvement,
            "improvement_threshold": 90.0,
            "individual_results": valid_results
        }
        
        if error_results:
            summary["errors"] = error_results
        
        return summary

def main():
    """Main function to run all tests"""
    print("Starting PDF Analyzer Testing Suite")
    print("=" * 50)
    
    # Initialize test runner
    runner = SimpleTestRunner()
    
    # Run all tests
    results = runner.run_all_tests()
    
    if not results:
        print("No test results obtained")
        return False
    
    # Generate summary report
    summary = runner.generate_summary_report(results)
    
    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY REPORT")
    print(f"{'='*50}")
    
    if "error" in summary:
        print(f"Error generating summary: {summary['error']}")
        return False
    
    print(f"Total Collections: {summary['total_collections']}")
    print(f"Successful Tests: {summary['successful_tests']}")
    print(f"Failed Tests: {summary['failed_tests']}")
    
    if summary['successful_tests'] > 0:
        print(f"Average Accuracy: {summary['average_accuracy']:.2f}%")
        print(f"Minimum Accuracy: {summary['min_accuracy']:.2f}%")
        print(f"Maximum Accuracy: {summary['max_accuracy']:.2f}%")
        print(f"Needs Improvement: {'Yes' if summary['needs_improvement'] else 'No'}")
        
        if summary['needs_improvement']:
            print(f"\n⚠️  Average accuracy ({summary['average_accuracy']:.2f}%) is below threshold (90%)")
            print("Consider running model optimization: python model_optimizer.py")
        else:
            print(f"\n✅ Average accuracy ({summary['average_accuracy']:.2f}%) meets the threshold!")
    
    # Save detailed report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"test_results_{timestamp}.json"
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"\nDetailed report saved to: {report_file}")
    except Exception as e:
        print(f"Error saving report: {e}")
    
    return summary['successful_tests'] > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
