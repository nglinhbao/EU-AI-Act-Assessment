import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from evaluation import evaluate

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
    # Combine the system description and the prompt for classification
    combined_input = f"System Description: {system_description}\n\nPrompt: {prompt}\n\nAnswer: "
    inputs = tokenizer(combined_input, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-process the response to extract the number
    answer = response.split("Answer: ")[-1].strip()
    
    # Ensure the answer is a digit between 1 and 5
    if answer.isdigit() and 1 <= int(answer) <= 5:
        return int(answer)
    else:
        return 3

def perform_classification(row, model, tokenizer, prompts_df, viz=True):
    # Here we create a more detailed input description using multiple columns
    input_description = f"{row['AI System Description']}. It uses {row['Input Data Type']}. The system functions include: {row['System Functions']}. Benefits of Commercial Use: {row['Benefits of Commercial Use']}. Assumptions/Consents Regarding Data Usage: {row['Assumptions/Consents Regarding Data Usage']}"
    
    # The prompt that the model will evaluate
    prompt = """
    Based on the description, assess the system on a scale of 1-5 based on a given question. 1 means absolutely no, and 5 means absolutely yes. 

    Here are some examples:

    Question: Is this AI system used for entertainment purposes, such as AI-enabled video games?
    Answer: 4

    Question: Does this AI system process sensitive personal data without user consent?
    Answer: 2

    Question: Is this AI system involved in biometric surveillance in public spaces?
    Answer: 5

    Now, based on the description provided, answer the following question (Only return a single number for the result. Do not leave answers blank):
    """    
    current_level = "High Risk"
    current_value = 0
    current_count = 0
    
    for index, prompt_row in prompts_df.iterrows():
        type = prompt_row['Type']
        prompt_question = prompt + prompt_row['Prompt']
        answer = query_llama(model, tokenizer, input_description, prompt_question)
        if viz:
            print(f"Response: {answer}")

        level = answer

        if current_level != type:
            if current_count > 0:
                result = float(current_value / current_count)
                if result > 3:
                    return current_level
            if type == "Minimal Risk":
                break
            current_level = type
            current_value = level
            current_count = 1
        else:
            current_value += level
            current_count += 1

    return "Minimal Risk"

def main():
    # txt_file_path = './input.txt'
    csv_file_path = './ai_risk_prompts.csv'
    model_name = "meta-llama/Llama-2-7b-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    prompts_df = pd.read_csv(csv_file_path)

    synthetic_dataset = './sample.csv'
    df = retrive_information_from_csv(synthetic_dataset)
    
    # Incorporate more columns into the input description and classification process
    df['Result'] = df.apply(lambda row: perform_classification(row, model, tokenizer, prompts_df), axis=1)
    
    df = evaluate(df)

    print(f"Accuracy: {df['Accuracy']:.2f}")

if __name__ == "__main__":
    main()
