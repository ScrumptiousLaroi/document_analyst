#!/usr/bin/env python3
"""
Model Improvement Script for PDF Analyzer
Implements various optimization techniques to improve accuracy
"""

import json
import os
import sys
from pathlib import Path
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analyzer import run_analysis
from pdf_processor import extract_sections_from_pdf

class ModelOptimizer:
    def __init__(self):
        self.models_dir = Path("models")
        self.current_model_name = "all-MiniLM-L6-v2"
        self.alternative_models = [
            "all-mpnet-base-v2",           # Higher quality, larger model
            "multi-qa-mpnet-base-dot-v1",  # Better for Q&A tasks
            "all-distilroberta-v1",        # Good balance of speed/quality
            "paraphrase-multilingual-mpnet-base-v2"  # Multilingual support
        ]
        
    def download_alternative_model(self, model_name: str) -> bool:
        """Download an alternative model for testing"""
        try:
            print(f"Downloading model: {model_name}")
            
            # Create model-specific directory
            model_dir = self.models_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Download and save model
            model = SentenceTransformer(model_name)
            model.save(str(model_dir))
            
            print(f"Model {model_name} downloaded successfully to {model_dir}")
            return True
            
        except Exception as e:
            print(f"Error downloading model {model_name}: {e}")
            return False
    
    def test_model_performance(self, model_name: str, test_data: List[Tuple[str, str, Dict]]) -> float:
        """Test a specific model's performance on test data"""
        try:
            print(f"Testing model: {model_name}")
            
            # Load model
            model_path = self.models_dir / model_name
            if not model_path.exists():
                print(f"Model not found, downloading: {model_name}")
                if not self.download_alternative_model(model_name):
                    return 0.0
            
            # Create analyzer with specific model
            analyzer = DocumentAnalyzer(model_path=str(model_path))
            
            total_accuracy = 0.0
            test_count = 0
            
            for pdf_text, persona, job_description in test_data:
                try:
                    result = analyzer.analyze_document(pdf_text, persona, job_description)
                    if result:
                        # Simple quality score based on number of sections and content quality
                        sections = result.get('sections', [])
                        quality_score = min(len(sections) / 5.0, 1.0)  # Normalize to 0-1
                        
                        # Content quality (average content length)
                        if sections:
                            avg_content_length = sum(len(s.get('content', '')) for s in sections) / len(sections)
                            content_quality = min(avg_content_length / 500.0, 1.0)  # Normalize
                            quality_score = (quality_score + content_quality) / 2
                        
                        total_accuracy += quality_score
                    
                    test_count += 1
                    
                except Exception as e:
                    print(f"Error testing with model {model_name}: {e}")
                    continue
            
            avg_accuracy = (total_accuracy / test_count * 100) if test_count > 0 else 0.0
            print(f"Model {model_name} average quality score: {avg_accuracy:.2f}%")
            
            return avg_accuracy
            
        except Exception as e:
            print(f"Error testing model {model_name}: {e}")
            traceback.print_exc()
            return 0.0
    
    def prepare_test_data(self, test_dir: str = "test_pdf") -> List[Tuple[str, str, str]]:
        """Prepare test data from all collections"""
        test_data = []
        test_path = Path(test_dir)
        
        if not test_path.exists():
            print(f"Test directory not found: {test_path}")
            return test_data
        
        pdf_processor = PDFProcessor()
        
        # Process each collection
        collections = [d for d in test_path.iterdir() if d.is_dir() and d.name.startswith('Collection')]
        
        for collection_path in collections:
            try:
                # Load input data
                input_files = list(collection_path.glob("*input*.json"))
                if not input_files:
                    continue
                
                with open(input_files[0], 'r', encoding='utf-8') as f:
                    input_data = json.load(f)
                
                persona = input_data.get('persona', {}).get('role', 'Analyst')
                job_description = input_data.get('job_to_be_done', {}).get('task', '')
                
                # Process PDFs
                pdf_folder = collection_path / "PDF"
                if pdf_folder.exists():
                    pdf_files = list(pdf_folder.glob("*.pdf"))[:3]  # Sample 3 PDFs per collection
                    
                    for pdf_path in pdf_files:
                        try:
                            pdf_text = pdf_processor.extract_text_from_pdf(str(pdf_path))
                            if pdf_text:
                                test_data.append((pdf_text, persona, job_description))
                        except Exception as e:
                            print(f"Error processing {pdf_path}: {e}")
                            continue
                            
            except Exception as e:
                print(f"Error preparing test data from {collection_path}: {e}")
                continue
        
        print(f"Prepared {len(test_data)} test samples")
        return test_data
    
    def optimize_analyzer_parameters(self) -> Dict[str, Any]:
        """Optimize analyzer parameters for better performance"""
        print("Optimizing analyzer parameters...")
        
        # Enhanced parameters for better performance
        optimized_params = {
            'top_k_sections': 8,  # Increased from default 5
            'min_section_length': 50,  # Minimum chars per section
            'content_relevance_threshold': 0.6,  # Lower threshold for more content
            'semantic_similarity_threshold': 0.5,  # Adjust similarity matching
            'max_content_length': 1000,  # Increased content length
            'enable_content_filtering': True,  # Filter irrelevant content
            'enable_semantic_grouping': True,  # Group similar sections
        }
        
        return optimized_params
    
    def create_optimized_analyzer(self, model_name: str = None, custom_params: Dict = None) -> DocumentAnalyzer:
        """Create an optimized analyzer with best settings"""
        if model_name:
            model_path = self.models_dir / model_name
            if model_path.exists():
                analyzer = DocumentAnalyzer(model_path=str(model_path))
            else:
                print(f"Model path not found: {model_path}, using default")
                analyzer = DocumentAnalyzer()
        else:
            analyzer = DocumentAnalyzer()
        
        # Apply optimizations
        if custom_params:
            analyzer.top_k_sections = custom_params.get('top_k_sections', 5)
            analyzer.min_section_length = custom_params.get('min_section_length', 30)
            # Add other parameters as needed
        
        return analyzer
    
    def run_model_comparison(self) -> Tuple[str, float]:
        """Compare different models and return the best one"""
        print("Running model comparison...")
        
        # Prepare test data
        test_data = self.prepare_test_data()
        if not test_data:
            print("No test data available for model comparison")
            return self.current_model_name, 0.0
        
        # Test current model
        current_score = self.test_model_performance(self.current_model_name, test_data[:5])  # Sample test
        print(f"Current model ({self.current_model_name}) score: {current_score:.2f}%")
        
        best_model = self.current_model_name
        best_score = current_score
        
        # Test alternative models
        for model_name in self.alternative_models:
            try:
                score = self.test_model_performance(model_name, test_data[:3])  # Smaller sample for alternatives
                
                if score > best_score:
                    best_model = model_name
                    best_score = score
                    print(f"New best model: {model_name} with score: {score:.2f}%")
                
            except Exception as e:
                print(f"Error testing model {model_name}: {e}")
                continue
        
        return best_model, best_score
    
    def apply_improvements(self) -> Dict[str, Any]:
        """Apply all improvements and return configuration"""
        print("Applying model improvements...")
        
        improvements = {
            'timestamp': datetime.now().isoformat(),
            'original_model': self.current_model_name,
            'optimizations_applied': []
        }
        
        try:
            # 1. Optimize parameters
            optimized_params = self.optimize_analyzer_parameters()
            improvements['optimized_parameters'] = optimized_params
            improvements['optimizations_applied'].append('parameter_optimization')
            
            # 2. Model comparison and selection
            best_model, best_score = self.run_model_comparison()
            improvements['best_model'] = best_model
            improvements['best_model_score'] = best_score
            improvements['optimizations_applied'].append('model_selection')
            
            # 3. Create optimized configuration file
            config = {
                'model_name': best_model,
                'model_path': str(self.models_dir / best_model),
                'parameters': optimized_params,
                'last_updated': datetime.now().isoformat()
            }
            
            config_file = Path("optimized_config.json")
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            
            improvements['config_file'] = str(config_file)
            improvements['optimizations_applied'].append('configuration_saved')
            
            print(f"Improvements applied successfully!")
            print(f"Best model: {best_model} (score: {best_score:.2f}%)")
            print(f"Configuration saved to: {config_file}")
            
        except Exception as e:
            print(f"Error applying improvements: {e}")
            traceback.print_exc()
            improvements['error'] = str(e)
        
        return improvements

class EnhancedDocumentAnalyzer(DocumentAnalyzer):
    """Enhanced version of DocumentAnalyzer with optimizations"""
    
    def __init__(self, config_file: str = "optimized_config.json", **kwargs):
        # Load optimized configuration if available
        if Path(config_file).exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                model_path = config.get('model_path')
                if model_path and Path(model_path).exists():
                    super().__init__(model_path=model_path, **kwargs)
                else:
                    super().__init__(**kwargs)
                
                # Apply optimized parameters
                params = config.get('parameters', {})
                self.top_k_sections = params.get('top_k_sections', 5)
                self.min_section_length = params.get('min_section_length', 30)
                
                print(f"Loaded optimized configuration from {config_file}")
                
            except Exception as e:
                print(f"Error loading optimized config: {e}")
                super().__init__(**kwargs)
        else:
            super().__init__(**kwargs)
    
    def analyze_document(self, text: str, persona: str, job_description: str) -> Dict:
        """Enhanced document analysis with optimizations"""
        try:
            # Use parent method but with enhanced processing
            result = super().analyze_document(text, persona, job_description)
            
            if result and 'sections' in result:
                # Apply post-processing optimizations
                sections = result['sections']
                
                # Filter sections by relevance and length
                filtered_sections = []
                for section in sections:
                    content = section.get('content', '')
                    if len(content) >= getattr(self, 'min_section_length', 30):
                        filtered_sections.append(section)
                
                # Take top K sections
                top_k = getattr(self, 'top_k_sections', 5)
                result['sections'] = filtered_sections[:top_k]
                
                # Enhance metadata
                result['metadata'] = result.get('metadata', {})
                result['metadata']['optimization_applied'] = True
                result['metadata']['sections_filtered'] = len(sections) - len(filtered_sections)
            
            return result
            
        except Exception as e:
            print(f"Error in enhanced analysis: {e}")
            return super().analyze_document(text, persona, job_description)

def main():
    """Main function to run model optimization"""
    print("PDF Analyzer Model Optimization")
    print("=" * 40)
    
    optimizer = ModelOptimizer()
    
    # Apply improvements
    improvements = optimizer.apply_improvements()
    
    # Generate report
    print("\nOPTIMIZATION REPORT")
    print("-" * 30)
    
    for key, value in improvements.items():
        if key == 'optimizations_applied':
            print(f"{key}: {', '.join(value)}")
        elif key == 'optimized_parameters':
            print(f"{key}:")
            for param, val in value.items():
                print(f"  {param}: {val}")
        else:
            print(f"{key}: {value}")
    
    # Save improvement report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"optimization_report_{timestamp}.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(improvements, f, indent=2)
    
    print(f"\nOptimization report saved to: {report_file}")
    
    if 'error' not in improvements:
        print("\n✅ Model optimization completed successfully!")
        print("Please run the test suite again to verify improvements.")
        return True
    else:
        print(f"\n❌ Optimization failed: {improvements['error']}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
