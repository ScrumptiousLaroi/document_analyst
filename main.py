import json
import os
from datetime import datetime
from analyzer import run_analysis

# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == "__main__":
    # Step 2: Define the input based on the specified format.
    # This would typically come from an API request or another source.
    input_json = {
    "challenge_info": {
        "challenge_id": "round_1b_002",
        "test_case_name": "travel_planner",
        "description": "France Travel"
    },
    "documents": [
        {
            "filename": "South of France - Cities.pdf",
            "title": "South of France - Cities"
        },
        {
            "filename": "South of France - Cuisine.pdf",
            "title": "South of France - Cuisine"
        },
        {
            "filename": "South of France - History.pdf",
            "title": "South of France - History"
        },
        {
            "filename": "South of France - Restaurants and Hotels.pdf",
            "title": "South of France - Restaurants and Hotels"
        },
        {
            "filename": "South of France - Things to Do.pdf",
            "title": "South of France - Things to Do"
        },
        {
            "filename": "South of France - Tips and Tricks.pdf",
            "title": "South of France - Tips and Tricks"
        },
        {
            "filename": "South of France - Traditions and Culture.pdf",
            "title": "South of France - Traditions and Culture"
        }
    ],
    "persona": {
        "role": "Travel Planner"
    },
    "job_to_be_done": {
        "task": "Plan a trip of 4 days for a group of 10 college friends."
    }
}

    # Step 3: Run the analysis by calling the main function from analyzer.py
    result = run_analysis(input_json)

    # Step 4: Save the final, formatted JSON output to a file
    if result:
        # Create output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"analysis_result_{timestamp}.json"
        output_path = os.path.join("output", output_filename)
        
        # Ensure output directory exists
        os.makedirs("output", exist_ok=True)
        
        # Save the result to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n--- OUTPUT SAVED ---")
        print(f"Analysis result saved to: {output_path}")
    else:
        print("\n--- NO OUTPUT ---")
        print("No result to save.")

