import pandas as pd
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import os

# Directly specify the AI system description file path
txt_file_path = './input.txt'

# Load the AI system description from the provided .txt file
def get_system_description():
    if not os.path.isfile(txt_file_path):
        print("File not found. Please provide a valid .txt file path.")
        return None
    
    try:
        with open(txt_file_path, 'r') as file:
            system_description = file.read()
        return system_description
    except Exception as e:
        print(f"Error reading the .txt file: {e}")
        return None

# Function to query LLaMA 2 using the system description and prompt
def query_llama(model, tokenizer, system_description, prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Combine the system description and the prompt for classification
    combined_input = f"System Description: {system_description}\n\nPrompt: {prompt}"
    inputs = tokenizer(combined_input, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Perform classification by reading the CSV from top to bottom
def perform_classification(model, tokenizer, system_description, prompts_df):
    # Loop through the CSV and check each prompt
    for index, row in prompts_df.iterrows():
        type = row['Type']
        prompt = row['Prompt']
        print(f"Prompt: {prompt}")
        answer = query_llama(model, tokenizer, system_description, prompt)
        print(f"Response: {answer}")
        
        # If the response contains 'yes', return the current type (risk category)
        if "yes" in answer.lower():
            print(f"Classified as: {type}")
            return type
    
    # If no 'yes' responses, return minimal risk by default
    return "Minimal Risk"

# Main function to handle AI system description and classification
def main():
    # Get the AI system description from the provided file
    system_description = get_system_description()
    if system_description is None:
        return
    
    # Load prompts from CSV file
    csv_file_path = './ai_risk_prompts.csv'  # Ensure this CSV exists with appropriate prompts
    prompts_df = pd.read_csv(csv_file_path)
    
    # Load the model and tokenizer for LLaMA 2
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    # Perform classification by processing the CSV row by row
    classification_status = perform_classification(model, tokenizer, system_description, prompts_df)

    print(f"The AI system is classified as: {classification_status}")

# Run the main function
if __name__ == "__main__":
    main()