import csv
import os
import sys
from utils.cleanDesc import clean_description

def add_cleaned_descriptions(input_csv_path, output_csv_path=None):
    if not output_csv_path:
        base, ext = os.path.splitext(input_csv_path)
        output_csv_path = f"{base}_cleaned{ext}"
    
    try:
        # Read the input CSV
        with open(input_csv_path, 'r', encoding='utf-8') as input_file:
            reader = csv.DictReader(input_file)
            fieldnames = reader.fieldnames + ['cleaned_desc']
            
            # Prepare for writing the output CSV
            with open(output_csv_path, 'w', encoding='utf-8', newline='') as output_file:
                writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                writer.writeheader()
                
                # Process each row
                for row in reader:
                    # Clean the description using our function
                    description = row.get('description', '')
                    cleaned_description = clean_description(description)
                    
                    # Add the cleaned description to the row
                    row['cleaned_desc'] = cleaned_description
                    
                    # Write the updated row
                    writer.writerow(row)
                    
        print(f"Successfully processed {input_csv_path}")
        print(f"Output saved to {output_csv_path}")
        
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # If run as a script, process the repositories.csv file
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        # Default path that works with the current directory structure
        input_path = 'datasets/repositories.csv'
        
        # Check if file exists at the default path
        if not os.path.exists(input_path):
            print(f"Warning: Default file not found at {input_path}")
            print("Please provide the path to repositories.csv as a command line argument")
            print("Example: python add_cleaned_desc.py path/to/repositories.csv")
            sys.exit(1)
    
    add_cleaned_descriptions(input_path) 