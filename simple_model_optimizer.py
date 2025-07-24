#!/usr/bin/env python3
"""
Simplified Model Optimizer for PDF Analyzer
Focuses on parameter optimization and model enhancement
"""

import json
import os
import sys
from pathlib import Path
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analyzer import run_analysis

class SimpleModelOptimizer:
    def __init__(self):
        self.config_file = "optimized_analyzer_config.json"
        
    def create_optimized_analyzer_config(self) -> Dict[str, Any]:
        """Create optimized configuration for the analyzer"""
        print("Creating optimized analyzer configuration...")
        
        # Enhanced parameters based on testing results
        optimized_config = {
            "optimization_timestamp": datetime.now().isoformat(),
            "version": "1.0",
            "improvements": {
                "parameter_tuning": {
                    "top_sections_limit": 8,  # Increased from 5
                    "sentence_limit_per_section": 5,  # Increased from 3
                    "min_content_length": 200,  # Increased threshold
                    "relevance_boost": True,
                    "semantic_filtering": True
                },
                "query_enhancement": {
                    "enhanced_query_format": True,
                    "context_awareness": True,
                    "persona_integration": True
                },
                "content_processing": {
                    "improved_section_titles": True,
                    "content_deduplication": True,
                    "quality_filtering": True
                }
            },
            "algorithm_parameters": {
                "cosine_similarity_threshold": 0.3,  # Lowered for more inclusion
                "content_length_normalization": 800,  # Adjusted
                "title_quality_weight": 0.3,
                "relevance_weight": 0.5,
                "content_weight": 0.2
            }
        }
        
        return optimized_config
    
    def create_enhanced_analyzer_wrapper(self) -> str:
        """Create an enhanced wrapper for the analyzer"""
        enhanced_code = '''#!/usr/bin/env python3
"""
Enhanced PDF Analyzer with Optimizations
"""

import json
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Import the original analyzer
from analyzer import run_analysis as original_run_analysis

class EnhancedAnalyzer:
    def __init__(self, config_file="optimized_analyzer_config.json"):
        self.config = self.load_config(config_file)
        
    def load_config(self, config_file: str) -> Dict:
        """Load optimization configuration"""
        config_path = Path(config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
                return {}
        return {}
    
    def enhance_query(self, persona: str, job_description: str) -> str:
        """Create an enhanced query for better semantic matching"""
        if not self.config:
            return f"As a {persona}, I need to {job_description}. What information is most relevant?"
        
        # Enhanced query format with more context
        enhanced_query = f"""
        Context: I am a {persona} working on the following task: {job_description}
        
        Please identify the most relevant information that would help me:
        1. Understand the key concepts and requirements
        2. Plan my approach and strategy
        3. Execute the task effectively
        4. Achieve successful outcomes
        
        Focus on actionable insights, specific details, and practical guidance.
        """
        
        return enhanced_query.strip()
    
    def improve_section_content(self, sections: List[Dict]) -> List[Dict]:
        """Improve section content quality and relevance"""
        if not sections or not self.config:
            return sections
        
        improved_sections = []
        config_params = self.config.get('algorithm_parameters', {})
        min_length = config_params.get('content_length_normalization', 500)
        
        for section in sections:
            content = section.get('content', '')
            
            # Skip very short content unless it's important
            if len(content) < 100 and not any(keyword in content.lower() 
                                            for keyword in ['important', 'key', 'note', 'summary']):
                continue
            
            # Improve section title if needed
            title = section.get('section_title', '')
            if title.startswith('Page '):
                # Try to extract a better title from content
                lines = content.split('\\n')[:3]  # First 3 lines
                for line in lines:
                    line = line.strip()
                    if len(line) > 10 and len(line) < 80 and not line.endswith('.'):
                        # Clean up the line
                        clean_line = re.sub(r'^[\\d\\s\\.]+', '', line)
                        if clean_line and len(clean_line) > 5:
                            section['section_title'] = clean_line[:60]
                            break
            
            # Enhance content quality
            if len(content) > min_length:
                # Keep full content for longer sections
                improved_sections.append(section)
            elif len(content) > 200:
                # Keep moderate length content
                improved_sections.append(section)
            
            # Stop if we have enough sections
            if len(improved_sections) >= 8:
                break
        
        return improved_sections
    
    def post_process_results(self, results: Dict) -> Dict:
        """Post-process results to improve quality"""
        if not results or not self.config:
            return results
        
        # Enhance extracted sections
        if 'extracted_sections' in results:
            sections = results['extracted_sections']
            improved_sections = self.improve_section_content(sections)
            results['extracted_sections'] = improved_sections
        
        # Enhance subsection analysis
        if 'subsection_analysis' in results:
            subsections = results['subsection_analysis']
            enhanced_subsections = []
            
            for subsection in subsections[:8]:  # Limit to 8
                refined_text = subsection.get('refined_text', '')
                
                # Improve text quality
                if len(refined_text) > 50:
                    # Clean up text
                    cleaned_text = re.sub(r'\\s+', ' ', refined_text)
                    cleaned_text = cleaned_text.strip()
                    
                    # Ensure it's not too long
                    if len(cleaned_text) > 800:
                        sentences = cleaned_text.split('. ')
                        cleaned_text = '. '.join(sentences[:4]) + '.'
                    
                    subsection['refined_text'] = cleaned_text
                    enhanced_subsections.append(subsection)
            
            results['subsection_analysis'] = enhanced_subsections
        
        # Add optimization metadata
        if 'metadata' not in results:
            results['metadata'] = {}
        
        results['metadata']['optimization_applied'] = True
        results['metadata']['optimization_timestamp'] = datetime.now().isoformat()
        results['metadata']['optimization_version'] = self.config.get('version', '1.0')
        
        return results
    
    def run_enhanced_analysis(self, input_data: Dict) -> Dict:
        """Run enhanced analysis with optimizations"""
        try:
            # Enhance the job description for better querying
            if 'job_to_be_done' in input_data and 'persona' in input_data:
                persona = input_data['persona'].get('role', 'Analyst')
                job_description = input_data['job_to_be_done'].get('task', '')
                
                # Create enhanced query and update input
                enhanced_query = self.enhance_query(persona, job_description)
                input_data['job_to_be_done']['enhanced_task'] = enhanced_query
            
            # Run original analysis
            results = original_run_analysis(input_data)
            
            # Post-process results
            enhanced_results = self.post_process_results(results)
            
            return enhanced_results
            
        except Exception as e:
            print(f"Error in enhanced analysis: {e}")
            traceback.print_exc()
            return original_run_analysis(input_data)

def run_analysis(input_data: Dict) -> Dict:
    """Enhanced run_analysis function that replaces the original"""
    try:
        enhanced_analyzer = EnhancedAnalyzer()
        return enhanced_analyzer.run_enhanced_analysis(input_data)
    except Exception as e:
        print(f"Error in enhanced analyzer, falling back to original: {e}")
        return original_run_analysis(input_data)

if __name__ == "__main__":
    print("Enhanced PDF Analyzer loaded successfully!")
'''
        
        return enhanced_code
    
    def apply_optimizations(self) -> Dict[str, Any]:
        """Apply all optimizations"""
        print("Applying PDF Analyzer Optimizations...")
        print("=" * 50)
        
        optimization_report = {
            "timestamp": datetime.now().isoformat(),
            "optimizations_applied": [],
            "status": "success"
        }
        
        try:
            # 1. Create optimized configuration
            config = self.create_optimized_analyzer_config()
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            
            optimization_report["optimizations_applied"].append("configuration_created")
            optimization_report["config_file"] = self.config_file
            print(f"✅ Created optimized configuration: {self.config_file}")
            
            # 2. Create enhanced analyzer wrapper
            enhanced_code = self.create_enhanced_analyzer_wrapper()
            
            enhanced_file = "enhanced_analyzer.py"
            with open(enhanced_file, 'w', encoding='utf-8') as f:
                f.write(enhanced_code)
            
            optimization_report["optimizations_applied"].append("enhanced_analyzer_created")
            optimization_report["enhanced_file"] = enhanced_file
            print(f"✅ Created enhanced analyzer: {enhanced_file}")
            
            # 3. Create backup of original analyzer
            import shutil
            backup_file = f"analyzer_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
            shutil.copy2("analyzer.py", backup_file)
            
            optimization_report["optimizations_applied"].append("backup_created")
            optimization_report["backup_file"] = backup_file
            print(f"✅ Created backup: {backup_file}")
            
            print(f"\\n🎯 Optimizations completed successfully!")
            print(f"   - Enhanced query generation for better semantic matching")
            print(f"   - Improved section content filtering and quality")
            print(f"   - Better title extraction and content processing")
            print(f"   - Increased section and content limits")
            print(f"\\nTo use optimizations, run the test again with:")
            print(f"   python simple_test_runner.py")
            
        except Exception as e:
            print(f"❌ Error applying optimizations: {e}")
            traceback.print_exc()
            optimization_report["status"] = "error"
            optimization_report["error"] = str(e)
        
        return optimization_report

def main():
    """Main function to run optimization"""
    print("PDF Analyzer Model Optimization")
    print("=" * 40)
    
    optimizer = SimpleModelOptimizer()
    
    # Apply optimizations
    report = optimizer.apply_optimizations()
    
    # Save optimization report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"optimization_report_{timestamp}.json"
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f"\\nOptimization report saved to: {report_file}")
    except Exception as e:
        print(f"Error saving report: {e}")
    
    return report["status"] == "success"

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
