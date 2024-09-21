import pandas as pd
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# Load prompts from the CSV file
csv_file_path = './ai_system_risk_prompts.csv'
prompts_df = pd.read_csv(csv_file_path)

# Define sections for classification
initial_risk_prompts = prompts_df[prompts_df['Step'] == 'Initial Risk Assessment']['Prompt'].tolist()
level_based_prompts = prompts_df[prompts_df['Step'] == 'Level-Based Risk Assessment']['Prompt'].tolist()
risk_categorization_prompts = prompts_df[prompts_df['Step'] == 'Risk Categorization']['Prompt'].tolist()
ongoing_monitoring_prompts = prompts_df[prompts_df['Step'] == 'Ongoing Monitoring']['Prompt'].tolist()

# Function to query LLaMA 2
def query_llama(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # GPU
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Perform classification by looping through initial and level-based assessment
def perform_classification(initial_risk_prompts, level_based_prompts):
    # Step 1: Initial Risk Assessment
    for prompt in initial_risk_prompts:
        print(f"Prompt: {prompt}")
        answer = query_llama(model, tokenizer, prompt)
        print(f"Response: {answer}")
        if "yes" in answer.lower():
            return "Unacceptable Risk"
    
    # Step 2: Level-Based Risk Assessment
    for prompt in level_based_prompts:
        print(f"Prompt: {prompt}")
        answer = query_llama(model, tokenizer, prompt)
        print(f"Response: {answer}")

    # If no unacceptable risk, return classification prompts
    return "Continue to Risk Categorization"

# Perform risk categorization
def categorize_result():
    # Go through the categorization and ongoing prompts
    final_results = {}
    for prompt in risk_categorization_prompts + ongoing_monitoring_prompts:
        print(f"Prompt: {prompt}")
        answer = query_llama(model, tokenizer, prompt)
        print(f"Response: {answer}")
        final_results[prompt] = answer
    return final_results

# Load the model and tokenizer for LLaMA 2
# This is a placeholder, assuming you have the model loaded already
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Step-by-step execution
classification_status = perform_classification(initial_risk_prompts, level_based_prompts)

if classification_status == "Unacceptable Risk":
    print("The AI system is classified as Unacceptable Risk.")
else:
    final_risk_results = categorize_result()
    print("Final Risk Results:")
    for prompt, answer in final_risk_results.items():
        print(f"{prompt}: {answer}")
