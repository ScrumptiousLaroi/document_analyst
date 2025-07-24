#!/usr/bin/env python3
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
                lines = content.split('\n')[:3]  # First 3 lines
                for line in lines:
                    line = line.strip()
                    if len(line) > 10 and len(line) < 80 and not line.endswith('.'):
                        # Clean up the line
                        clean_line = re.sub(r'^[\d\s\.]+', '', line)
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
                    cleaned_text = re.sub(r'\s+', ' ', refined_text)
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
