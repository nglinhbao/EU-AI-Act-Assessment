import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
# from evaluation import evaluate
import re
from fairness_scoring import evaluate_app_fairness, load_models

def get_system_description(txt_file_path):
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

def retrive_information_from_csv(file_name):
    try:
        df = pd.read_csv(file_name)
        return df
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None

def query_llama(model, tokenizer, system_description, prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    combined_input = f"System Description: {system_description}\n\nPrompt: {prompt}"
    inputs = tokenizer(combined_input, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:        
        # Extract the second "Answer: "
        answer_parts = response.split("Answer: ")
        if len(answer_parts) < 3:  # Ensure we have at least two answers
            return {"score": 0.5, "reasoning": "Failed to find second answer"}
        
        answer_text = answer_parts[2].strip().split()[0]  # Get the first word after second "Answer: "
        # Map answer to score
        if answer_text.lower() in ["yes", " yes", "yes.", " yes."]:
            score = 1
        elif answer_text.lower() in ["no", " no", "no.", " no."]:
            score = 0
        else:
            score = 0.5  # Default to Neutral if answer is unrecognized
            
        # Extract the second "Reasoning: "
        reasoning_parts = response.split("Reasoning: ")
        if len(reasoning_parts) < 3:  # Ensure we have at least two reasonings
            reasoning = answer_parts[2].strip()
        else:
            reasoning = reasoning_parts[2].split("\n")[0].strip()  # Get the first line after second "Reasoning: "
        
        return {
            "score": score,
            "reasoning": reasoning
        }
        
    except Exception as e:
        print(f"Error parsing response: {e}")  # Debug print
        return {
            "score": 0.5,  # Default to Neutral in case of an error
            "reasoning": "Error occurred while parsing response."
        }

def check_inconsistency(row, model, tokenizer):
    difference_analysis = row['Difference Analysis']

    base_prompt = """
        Are Promised Features and User Reviews inconsistent?

        Answer the question only with 'Yes' or 'No'

        Template:
        Answer: [Yes/No]
        """

    full_prompt = difference_analysis + base_prompt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Use regex to extract the answer after "Answer:"
    match = re.search(r"Answer:\s*(Yes|No)", response, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower() == "yes"
    else:
        print("Unexpected model response:", response)
        return None

def perform_classification(row, model, tokenizer, prompts_df, viz=True):
    input_description = (
        f"{row['Full Description']}. "
        f"Additional app information: {row['App Info Modal']}. "
        f"Data shared with third parties: {row['Shared Data']}. "
        f"Data collected by the app: {row['Collected Data']}. "
        f"Security practices: {row['Security Practices']}."
    )

    word2vec_model, word2vec_words, classifier_model, use_model = load_models()

    if check_inconsistency(row, model, tokenizer):
        if evaluate_app_fairness(row["App Name"], word2vec_model, word2vec_words, classifier_model, use_model) > 0.4:
            input_description += f"User reviews: {row['User Review Analysis']}."
    
    base_prompt = """
        Answer the question with 'Yes' or 'No'
        (only answer the main question and do not split the question)
        
        Template:
        Answer: [Yes/No]
        Reasoning: [reasons why you give the score in 1 short paragraph]
    """
    
    score_sum = 0
    prompt_count = 0
    current_reasoning = ""
    
    for index, prompt_row in prompts_df.iterrows():
        risk_type = prompt_row['Type']
        full_prompt = base_prompt + prompt_row['Prompt']
        response = query_llama(model, tokenizer, input_description, full_prompt)
        
        if viz:
            print(f"Risk Type: {risk_type}")
            print(f"Prompt: {prompt_row['Prompt']}")
            print(f"Score: {response['score']}")
            print(f"Reasoning: {response['reasoning']}\n")
            
        score_sum += response['score']
        prompt_count += 1
        current_reasoning += response['reasoning']  # Keep the last reasoning
    
        if prompt_count > 0:
            confidence_score = score_sum / prompt_count
            if response['score'] == 1:
                return {
                    'risk_type': risk_type,
                    'confidence score': confidence_score,
                    'reasoning': current_reasoning
                }
            
        if risk_type == "Minimal Risk":
            break
    
    return {
        'risk_type': "Minimal Risk",
        'confidence score': confidence_score if 'confidence_score' in locals() else 0,
        'reasoning': current_reasoning if 'current_reasoning' in locals() else "No specific risks identified."
    }

def main():
    csv_file_path = '/content/drive/MyDrive/EU-AI-Act/datasets/ai_risk_prompts.csv'
    model_name = "TheBloke/Nous-Hermes-Llama2-GPTQ"
    # model_name = "meta-llama/Llama-2-7b-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    synthetic_dataset = '/content/drive/MyDrive/EU-AI-Act/datasets/app_reviews_analysis.csv'
    df = retrive_information_from_csv(synthetic_dataset)

    prompts_df = pd.read_csv("/content/drive/MyDrive/EU-AI-Act/datasets/ai_risk_prompts.csv")
    
    # Apply classification and extract results
    results = df.apply(lambda row: perform_classification(row, model, tokenizer, prompts_df), axis=1)
    
    # Assign risk type, score, and reasoning to separate columns
    df['Predicted System Type'] = results.apply(lambda x: x['risk_type'])
    df['Reason'] = results.apply(lambda x: x['reasoning'])

    # df = evaluate(df)
    #a

    df.to_csv('/content/drive/MyDrive/EU-AI-Act/datasets/results.csv', index=False)

    # print(f"Accuracy: {df['Accuracy']:.2f}")
    # print(f"Fairness Score: {df['Fairness Score']:.2f}")

if __name__ == "__main__":
    main()