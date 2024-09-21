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

# Perform classification by looping through initial and level-based assessment
def perform_classification(model, tokenizer, system_description, initial_risk_prompts, level_based_prompts):
    # Step 1: Initial Risk Assessment
    for prompt in initial_risk_prompts:
        print(f"Prompt: {prompt}")
        answer = query_llama(model, tokenizer, system_description, prompt)
        print(f"Response: {answer}")
        if "yes" in answer.lower():
            return "Unacceptable Risk"
    
    # Step 2: Level-Based Risk Assessment
    for prompt in level_based_prompts:
        print(f"Prompt: {prompt}")
        answer = query_llama(model, tokenizer, system_description, prompt)
        print(f"Response: {answer}")

    # If no unacceptable risk, return classification prompts
    return "Continue to Risk Categorization"

# Perform risk categorization
def categorize_result(model, tokenizer, system_description, risk_categorization_prompts, ongoing_monitoring_prompts):
    # Go through the categorization and ongoing prompts
    final_results = {}
    for prompt in risk_categorization_prompts + ongoing_monitoring_prompts:
        print(f"Prompt: {prompt}")
        answer = query_llama(model, tokenizer, system_description, prompt)
        print(f"Response: {answer}")
        final_results[prompt] = answer
    return final_results

# Main function to handle AI system description and classification
def main():
    # Get the AI system description from the provided file
    system_description = get_system_description()
    if system_description is None:
        return
    
    # Load prompts from CSV file
    csv_file_path = './ai_system_risk_prompts.csv'  # Ensure this CSV exists with appropriate prompts
    prompts_df = pd.read_csv(csv_file_path)
    
    # Define sections for classification
    initial_risk_prompts = prompts_df[prompts_df['Step'] == 'Initial Risk Assessment']['Prompt'].tolist()
    level_based_prompts = prompts_df[prompts_df['Step'] == 'Level-Based Risk Assessment']['Prompt'].tolist()
    risk_categorization_prompts = prompts_df[prompts_df['Step'] == 'Risk Categorization']['Prompt'].tolist()
    ongoing_monitoring_prompts = prompts_df[prompts_df['Step'] == 'Ongoing Monitoring']['Prompt'].tolist()

    # Load the model and tokenizer for LLaMA 2
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    # Perform classification
    classification_status = perform_classification(model, tokenizer, system_description, initial_risk_prompts, level_based_prompts)

    if classification_status == "Unacceptable Risk":
        print("The AI system is classified as Unacceptable Risk.")
    else:
        final_risk_results = categorize_result(model, tokenizer, system_description, risk_categorization_prompts, ongoing_monitoring_prompts)
        print("Final Risk Results:")
        for prompt, answer in final_risk_results.items():
            print(f"{prompt}: {answer}")

# Run the main function
if __name__ == "__main__":
    main()
