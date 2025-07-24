#!/usr/bin/env python3
"""
Fixed Enhanced PDF Analyzer with Optimizations
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

class FixedEnhancedAnalyzer:
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
        # Create a more detailed and context-rich query
        enhanced_query = f"""
        As a {persona}, I am tasked with: {job_description}
        
        I need to identify the most relevant and actionable information that will help me:
        - Understand key requirements and important details
        - Make informed decisions and create effective plans
        - Execute tasks successfully with practical guidance
        - Access specific recommendations and expert insights
        
        Please prioritize content that is directly applicable to my role and objectives.
        """
        
        return enhanced_query.strip()
    
    def improve_section_titles(self, sections: List[Dict]) -> List[Dict]:
        """Improve section titles without filtering out sections"""
        if not sections:
            return sections
        
        improved_sections = []
        
        for section in sections:
            # Create a copy to avoid modifying the original
            improved_section = section.copy()
            
            title = section.get('section_title', '')
            content = section.get('content', '')
            
            # Only improve titles that are generic page references
            if title.startswith('Page ') and content:
                # Try to extract a better title from content
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                
                for line in lines[:5]:  # Check first 5 lines
                    # Clean up the line
                    clean_line = re.sub(r'^\d+[\.\s]*', '', line)  # Remove leading numbers
                    clean_line = re.sub(r'^Page\s+\d+', '', clean_line, flags=re.IGNORECASE)
                    clean_line = clean_line.strip()
                    
                    # Check if this looks like a good title
                    if (10 <= len(clean_line) <= 80 and 
                        not clean_line.endswith('.') and 
                        not clean_line.startswith('http') and
                        clean_line.count(' ') <= 12):
                        
                        improved_section['section_title'] = clean_line[:60]
                        break
            
            improved_sections.append(improved_section)
        
        return improved_sections
    
    def enhance_subsection_content(self, subsections: List[Dict]) -> List[Dict]:
        """Enhance subsection content quality"""
        if not subsections:
            return subsections
        
        enhanced_subsections = []
        
        for subsection in subsections:
            enhanced_sub = subsection.copy()
            refined_text = subsection.get('refined_text', '')
            
            if refined_text:
                # Clean up text formatting
                cleaned_text = re.sub(r'\s+', ' ', refined_text)  # Normalize whitespace
                cleaned_text = cleaned_text.strip()
                
                # Ensure reasonable length (not too long)
                if len(cleaned_text) > 1000:
                    # Split into sentences and take the first several
                    sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
                    # Take enough sentences to get around 800 characters
                    truncated = ""
                    for sentence in sentences:
                        if len(truncated + sentence) > 800:
                            break
                        truncated += sentence + " "
                    cleaned_text = truncated.strip()
                
                enhanced_sub['refined_text'] = cleaned_text
            
            enhanced_subsections.append(enhanced_sub)
        
        return enhanced_subsections
    
    def post_process_results(self, results: Dict) -> Dict:
        """Post-process results to improve quality without breaking structure"""
        if not results:
            return results
        
        # Create a copy to avoid modifying the original
        enhanced_results = results.copy()
        
        # Improve extracted sections titles (but keep all sections)
        if 'extracted_sections' in results:
            sections = results['extracted_sections']
            if sections:  # Only process if there are sections
                improved_sections = self.improve_section_titles(sections)
                enhanced_results['extracted_sections'] = improved_sections
        
        # Enhance subsection analysis content
        if 'subsection_analysis' in results:
            subsections = results['subsection_analysis']
            if subsections:  # Only process if there are subsections
                enhanced_subsections = self.enhance_subsection_content(subsections)
                enhanced_results['subsection_analysis'] = enhanced_subsections
        
        # Add optimization metadata
        if 'metadata' not in enhanced_results:
            enhanced_results['metadata'] = {}
        
        enhanced_results['metadata']['optimization_applied'] = True
        enhanced_results['metadata']['optimization_timestamp'] = datetime.now().isoformat()
        enhanced_results['metadata']['optimization_version'] = "2.0_fixed"
        
        return enhanced_results
    
    def run_enhanced_analysis(self, input_data: Dict) -> Dict:
        """Run enhanced analysis with optimizations"""
        try:
            # Enhance the query for better semantic matching
            if 'job_to_be_done' in input_data and 'persona' in input_data:
                persona = input_data['persona'].get('role', 'Analyst')
                job_description = input_data['job_to_be_done'].get('task', '')
                
                # Create enhanced query and store it (but don't modify the original task)
                enhanced_query = self.enhance_query(persona, job_description)
                
                # Create a modified copy of input_data for processing
                enhanced_input = input_data.copy()
                enhanced_input['job_to_be_done'] = input_data['job_to_be_done'].copy()
                enhanced_input['job_to_be_done']['enhanced_task'] = enhanced_query
                
                # Use the enhanced query for analysis by temporarily modifying the task
                original_task = enhanced_input['job_to_be_done']['task']
                enhanced_input['job_to_be_done']['task'] = enhanced_query
                
                # Run analysis with enhanced query
                results = original_run_analysis(enhanced_input)
                
                # Restore original task in results metadata
                if results and 'metadata' in results:
                    results['metadata']['job_to_be_done'] = original_task
            else:
                # Run with original input if no persona/job info
                results = original_run_analysis(input_data)
            
            # Post-process results for improvements
            enhanced_results = self.post_process_results(results)
            
            return enhanced_results
            
        except Exception as e:
            print(f"Error in enhanced analysis, falling back to original: {e}")
            import traceback
            traceback.print_exc()
            return original_run_analysis(input_data)

def run_analysis(input_data: Dict) -> Dict:
    """Enhanced run_analysis function that replaces the original"""
    try:
        enhanced_analyzer = FixedEnhancedAnalyzer()
        return enhanced_analyzer.run_enhanced_analysis(input_data)
    except Exception as e:
        print(f"Error in enhanced analyzer, falling back to original: {e}")
        return original_run_analysis(input_data)

if __name__ == "__main__":
    print("Fixed Enhanced PDF Analyzer loaded successfully!")
