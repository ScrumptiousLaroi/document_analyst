#!/usr/bin/env python3
"""
Comprehensive Test Runner for PDF Analyzer
Runs tests on multiple collections and calculates accuracy metrics
"""

import json
import os
import sys
from pathlib import Path
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Any
import difflib
from dataclasses import dataclass

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analyzer import run_analysis
from pdf_processor import extract_sections_from_pdf

@dataclass
class TestResult:
    collection_name: str
    accuracy_score: float
    total_tests: int
    passed_tests: int
    failed_tests: int
    detailed_results: Dict[str, Any]
    execution_time: float

class TestRunner:
    def __init__(self, test_directory: str = "test_pdf"):
        self.test_directory = Path(test_directory)
        self.results = []
        
    def load_input_data(self, collection_path: Path) -> Dict:
        """Load input JSON data for a collection"""
        input_files = list(collection_path.glob("*input*.json"))
        if not input_files:
            raise FileNotFoundError(f"No input JSON found in {collection_path}")
        
        with open(input_files[0], 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_expected_output(self, collection_path: Path) -> Dict:
        """Load expected output JSON for a collection"""
        output_files = list(collection_path.glob("*output*.json"))
        if not output_files:
            raise FileNotFoundError(f"No output JSON found in {collection_path}")
        
        with open(output_files[0], 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def run_analyzer_on_collection(self, collection_path: Path) -> Dict:
        """Run the analyzer on a collection and return results"""
        try:
            # Load input data
            input_data = self.load_input_data(collection_path)
            
            # Check if PDF folder exists and has PDFs
            pdf_folder = collection_path / "PDF"
            if not pdf_folder.exists():
                print(f"Warning: No PDF folder found in {collection_path}")
                return {}
            
            pdf_files = list(pdf_folder.glob("*.pdf"))
            if not pdf_files:
                print(f"Warning: No PDF files found in {pdf_folder}")
                return {}
            
            # Update document paths in input_data to point to actual PDFs
            input_data["documents"] = []
            for pdf_file in pdf_files:
                input_data["documents"].append({
                    "filename": str(pdf_file),
                    "type": "pdf"
                })
            
            # Create a temporary copy of documents in the expected location
            temp_docs_dir = Path("documents")
            temp_docs_dir.mkdir(exist_ok=True)
            
            # Copy PDFs to documents directory temporarily
            for pdf_file in pdf_files:
                temp_pdf_path = temp_docs_dir / pdf_file.name
                if not temp_pdf_path.exists():
                    import shutil
                    shutil.copy2(pdf_file, temp_pdf_path)
                
                # Update input data to use just filename
                for doc in input_data["documents"]:
                    if doc["filename"] == str(pdf_file):
                        doc["filename"] = pdf_file.name
            
            # Run the analysis
            result = run_analysis(input_data)
            
            # Clean up temporary files
            for pdf_file in pdf_files:
                temp_pdf_path = temp_docs_dir / pdf_file.name
                if temp_pdf_path.exists():
                    temp_pdf_path.unlink()
            
            return result
            
        except Exception as e:
            print(f"Error running analyzer on {collection_path}: {e}")
            traceback.print_exc()
            return {}
    
    def calculate_accuracy(self, expected: Dict, actual: Dict) -> Tuple[float, Dict]:
        """Calculate accuracy between expected and actual results"""
        detailed_results = {
            'metadata_accuracy': 0.0,
            'sections_accuracy': 0.0,
            'subsections_accuracy': 0.0,
            'overall_structure_accuracy': 0.0
        }
        
        total_score = 0.0
        max_score = 4.0  # 4 categories
        
        try:
            # 1. Metadata accuracy
            expected_meta = expected.get('metadata', {})
            actual_meta = actual.get('metadata', {})
            
            meta_score = 0.0
            if expected_meta.get('persona') == actual_meta.get('persona'):
                meta_score += 0.5
            if expected_meta.get('job_to_be_done') == actual_meta.get('job_to_be_done'):
                meta_score += 0.5
            
            detailed_results['metadata_accuracy'] = meta_score
            total_score += meta_score
            
            # 2. Extracted sections accuracy
            expected_sections = expected.get('extracted_sections', [])
            actual_sections = actual.get('extracted_sections', [])
            
            sections_score = 0.0
            if len(actual_sections) > 0 and len(expected_sections) > 0:
                # Compare document coverage
                expected_docs = set(s.get('document', '') for s in expected_sections)
                actual_docs = set(s.get('document', '') for s in actual_sections)
                doc_overlap = len(expected_docs.intersection(actual_docs)) / len(expected_docs.union(actual_docs)) if expected_docs.union(actual_docs) else 0
                
                # Compare section titles (semantic similarity)
                expected_titles = [s.get('section_title', '') for s in expected_sections]
                actual_titles = [s.get('section_title', '') for s in actual_sections]
                
                title_similarity = self.calculate_text_similarity(expected_titles, actual_titles)
                
                sections_score = (doc_overlap + title_similarity) / 2
            
            detailed_results['sections_accuracy'] = sections_score
            total_score += sections_score
            
            # 3. Subsection analysis accuracy
            expected_subsections = expected.get('subsection_analysis', [])
            actual_subsections = actual.get('subsection_analysis', [])
            
            subsections_score = 0.0
            if len(actual_subsections) > 0 and len(expected_subsections) > 0:
                # Compare content similarity
                expected_content = [s.get('refined_text', '') for s in expected_subsections]
                actual_content = [s.get('refined_text', '') for s in actual_subsections]
                
                content_similarity = self.calculate_text_similarity(expected_content, actual_content)
                subsections_score = content_similarity
            
            detailed_results['subsections_accuracy'] = subsections_score
            total_score += subsections_score
            
            # 4. Overall structure accuracy
            structure_score = 0.0
            if 'metadata' in actual and 'extracted_sections' in actual and 'subsection_analysis' in actual:
                structure_score = 1.0
            
            detailed_results['overall_structure_accuracy'] = structure_score
            total_score += structure_score
            
            # Calculate overall accuracy percentage
            overall_accuracy = (total_score / max_score) * 100
            
            return overall_accuracy, detailed_results
            
        except Exception as e:
            print(f"Error calculating accuracy: {e}")
            return 0.0, detailed_results
    
    def calculate_text_similarity(self, expected_texts: List[str], actual_texts: List[str]) -> float:
        """Calculate similarity between two lists of texts"""
        if not expected_texts or not actual_texts:
            return 0.0
        
        similarities = []
        for expected_text in expected_texts:
            best_similarity = 0.0
            for actual_text in actual_texts:
                # Use difflib to calculate similarity
                similarity = difflib.SequenceMatcher(None, expected_text.lower(), actual_text.lower()).ratio()
                best_similarity = max(best_similarity, similarity)
            similarities.append(best_similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def run_single_collection_test(self, collection_path: Path) -> TestResult:
        """Run test on a single collection"""
        collection_name = collection_path.name
        print(f"\n{'='*50}")
        print(f"Testing Collection: {collection_name}")
        print(f"{'='*50}")
        
        start_time = datetime.now()
        
        try:
            # Find input and output files
            input_files = list(collection_path.glob("*input*.json"))
            output_files = list(collection_path.glob("*output*.json"))
            
            if not input_files:
                raise FileNotFoundError(f"No input JSON file found in {collection_path}")
            if not output_files:
                raise FileNotFoundError(f"No output JSON file found in {collection_path}")
            
            input_file = input_files[0]
            output_file = output_files[0]
            
            print(f"Input file: {input_file.name}")
            print(f"Output file: {output_file.name}")
            
            # Load expected input and output
            input_data = self.load_json_file(input_file)
            expected_output = self.load_json_file(output_file)
            
            if not input_data or not expected_output:
                raise ValueError("Failed to load input or output JSON files")
            
            # Run analyzer
            print("Running analyzer...")
            actual_output = self.run_analyzer_on_collection(collection_path, input_data)
            
            if not actual_output:
                raise ValueError("Analyzer failed to produce output")
            
            # Calculate accuracy
            print("Calculating accuracy...")
            accuracy, detailed_results = self.calculate_accuracy(expected_output, actual_output)
            
            # Create test result
            execution_time = (datetime.now() - start_time).total_seconds()
            
            test_result = TestResult(
                collection_name=collection_name,
                accuracy_score=accuracy,
                total_tests=1,
                passed_tests=1 if accuracy >= 90 else 0,
                failed_tests=0 if accuracy >= 90 else 1,
                detailed_results=detailed_results,
                execution_time=execution_time
            )
            
            # Save actual output for comparison
            output_dir = collection_path / "actual_output"
            output_dir.mkdir(exist_ok=True)
            
            actual_output_file = output_dir / f"actual_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(actual_output_file, 'w', encoding='utf-8') as f:
                json.dump(actual_output, f, indent=2, ensure_ascii=False)
            
            print(f"Actual output saved to: {actual_output_file}")
            
            return test_result
            
        except Exception as e:
            print(f"Error testing collection {collection_name}: {e}")
            traceback.print_exc()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            return TestResult(
                collection_name=collection_name,
                accuracy_score=0.0,
                total_tests=1,
                passed_tests=0,
                failed_tests=1,
                detailed_results={'error': str(e)},
                execution_time=execution_time
            )
    
    def run_all_tests(self) -> List[TestResult]:
        """Run tests on all collections"""
        print("Starting comprehensive test run...")
        print(f"Test directory: {self.test_dir}")
        
        if not self.test_dir.exists():
            print(f"Error: Test directory not found: {self.test_dir}")
            return []
        
        # Find all collection directories
        collections = [d for d in self.test_dir.iterdir() if d.is_dir() and d.name.startswith('Collection')]
        collections.sort()
        
        if not collections:
            print("No collection directories found!")
            return []
        
        print(f"Found {len(collections)} collections to test")
        
        results = []
        for collection_path in collections:
            result = self.run_single_collection_test(collection_path)
            results.append(result)
            
            # Print immediate results
            print(f"\nCollection: {result.collection_name}")
            print(f"Accuracy: {result.accuracy_score:.2f}%")
            print(f"Status: {'PASS' if result.accuracy_score >= 90 else 'FAIL'}")
            print(f"Execution time: {result.execution_time:.2f}s")
        
        self.results = results
        return results
    
    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        if not self.results:
            return "No test results available"
        
        report = []
        report.append("PDF ANALYZER TEST REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        total_collections = len(self.results)
        passed_collections = sum(1 for r in self.results if r.accuracy_score >= 90)
        failed_collections = total_collections - passed_collections
        avg_accuracy = sum(r.accuracy_score for r in self.results) / total_collections if total_collections > 0 else 0
        
        report.append("SUMMARY")
        report.append("-" * 20)
        report.append(f"Total Collections Tested: {total_collections}")
        report.append(f"Passed (≥90% accuracy): {passed_collections}")
        report.append(f"Failed (<90% accuracy): {failed_collections}")
        report.append(f"Average Accuracy: {avg_accuracy:.2f}%")
        report.append(f"Overall Success Rate: {(passed_collections/total_collections)*100:.1f}%")
        report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS")
        report.append("-" * 30)
        
        for result in self.results:
            report.append(f"\nCollection: {result.collection_name}")
            report.append(f"  Accuracy: {result.accuracy_score:.2f}%")
            report.append(f"  Status: {'PASS ✓' if result.accuracy_score >= 90 else 'FAIL ✗'}")
            report.append(f"  Execution Time: {result.execution_time:.2f}s")
            
            if 'error' not in result.detailed_results:
                report.append("  Detailed Metrics:")
                for metric, value in result.detailed_results.items():
                    report.append(f"    {metric}: {value:.2f}")
            else:
                report.append(f"  Error: {result.detailed_results['error']}")
        
        # Recommendations
        report.append("\n" + "RECOMMENDATIONS")
        report.append("-" * 30)
        
        if avg_accuracy < 90:
            report.append("⚠️  Overall accuracy is below 90%. Consider:")
            report.append("   1. Improving text extraction quality")
            report.append("   2. Enhancing semantic analysis algorithms")
            report.append("   3. Fine-tuning the similarity matching")
            report.append("   4. Upgrading to a more advanced language model")
        else:
            report.append("✅ Excellent performance! All collections meet accuracy requirements.")
        
        return "\n".join(report)
    
    def save_report(self, filename: str = None) -> str:
        """Save test report to file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"test_report_{timestamp}.txt"
        
        report_content = self.generate_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return filename

def main():
    """Main function to run tests"""
    print("PDF Analyzer Test Runner")
    print("=" * 30)
    
    # Initialize test runner
    runner = TestRunner()
    
    # Run all tests
    results = runner.run_all_tests()
    
    if not results:
        print("No tests were executed!")
        return
    
    # Generate and display report
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    
    report = runner.generate_report()
    print(report)
    
    # Save report
    report_file = runner.save_report()
    print(f"\nDetailed report saved to: {report_file}")
    
    # Check if improvements are needed
    avg_accuracy = sum(r.accuracy_score for r in results) / len(results)
    if avg_accuracy < 90:
        print(f"\n⚠️  ATTENTION: Average accuracy ({avg_accuracy:.2f}%) is below 90%")
        print("Consider running the model improvement script...")
        return False
    else:
        print(f"\n✅ SUCCESS: Average accuracy ({avg_accuracy:.2f}%) meets requirements!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
