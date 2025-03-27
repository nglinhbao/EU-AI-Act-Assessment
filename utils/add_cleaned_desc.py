import csv
import os
import sys
from cleanDesc import clean_description

def add_cleaned_descriptions(input_csv_path, output_csv_path=None):
    """
    Reads a CSV file containing repository data, cleans the descriptions,
    and writes a new CSV file with an additional 'cleaned_desc' column.
    
    Args:
        input_csv_path (str): Path to the input CSV file
        output_csv_path (str, optional): Path for the output CSV file. If None, 
                                         will use input_path with '_cleaned' suffix
    """
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
        # Default path (relative to the script location)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        input_path = os.path.join(parent_dir, 'datasets', 'repositories.csv')
    
    add_cleaned_descriptions(input_path) 