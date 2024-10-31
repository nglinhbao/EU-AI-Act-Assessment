import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
# from evaluation import evaluate
import re

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

def get_vector_db(risk_type, db_dir='./vector_db'):
    """
    Load the vector database for a specific risk type with safe deserialization.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    safe_name = risk_type.lower().replace(' ', '_')
    db_path = os.path.join(db_dir, safe_name)
    
    if not os.path.exists(db_path):
        raise ValueError(f"No vector database found for {risk_type}")
    
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

def get_relevant_prompts(system_description, risk_type, k):
    """
    Retrieve k most relevant prompts from the vector database for a given risk type.
    """
    try:
        db = get_vector_db(risk_type)
        results = db.similarity_search_with_score(system_description, k=k)
        # Extract just the prompts from the results
        prompts = [doc.page_content for doc, _ in results]
        return prompts
    except Exception as e:
        print(f"Error retrieving prompts for {risk_type}: {e}")
        return []

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

def perform_classification(row, model, tokenizer, viz=True):
    # Create detailed input description
    input_description = f"{row['AI System Description']}. It uses {row['Input Data Type']}. The system functions include: {row['System Functions']}. Benefits of Commercial Use: {row['Benefits of Commercial Use']}. Assumptions/Consents Regarding Data Usage: {row['Assumptions/Consents Regarding Data Usage']}"
    
    base_prompt = """
        Answer the question with 'Yes' or 'No'
        (only answer the main question and do not split the question)
        
        Template:
        Answer: [Yes/No]
        Reasoning: [reasons why you give the score in 1 short paragraph]
    """

    risk_types = ["Unacceptable Risk", "High Risk", "Limited Risk", "Minimal Risk"]
    
    for risk_type in risk_types:
        # Get relevant prompts for this risk type based on system description
        prompts = get_relevant_prompts(input_description, risk_type, 4)
        
        if not prompts:  # If no prompts found, skip this risk type
            continue
            
        score_sum = 0
        prompt_count = 0
        current_reasoning = ""
        
        for prompt_text in prompts:
            full_prompt = base_prompt + prompt_text
            response = query_llama(model, tokenizer, input_description, full_prompt)
            
            if viz:
                print(f"Risk Type: {risk_type}")
                print(f"Prompt: {prompt_text}")
                print(f"App description: {input_description}")
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
                        'confidence_score': confidence_score,
                        'reasoning': current_reasoning
                    }
                
            if risk_type == "Minimal Risk":
                break
    
    return {
        'risk_type': "Minimal Risk",
        'confidence_score': confidence_score if 'confidence_score' in locals() else 0,
        'reasoning': current_reasoning if 'current_reasoning' in locals() else "No specific risks identified."
    }

def main():
    csv_file_path = './datasets/ai_risk_prompts.csv'
    model_name = "TheBloke/Nous-Hermes-Llama2-GPTQ"
    # model_name = "meta-llama/Llama-2-7b-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    synthetic_dataset = './datasets/small_sample.csv'
    df = retrive_information_from_csv(synthetic_dataset)
    
    # Apply classification and extract results
    results = df.apply(lambda row: perform_classification(row, model, tokenizer), axis=1)
    
    # Assign risk type, score, and reasoning to separate columns
    df['Confidence Score'] = results.apply(lambda x: x['confidence_score'])
    df['Predicted System Type'] = results.apply(lambda x: x['risk_type'])
    df['Reason'] = results.apply(lambda x: x['reasoning'])

    # df = evaluate(df)

    df.to_csv('datasets/results.csv', index=False)

    # print(f"Accuracy: {df['Accuracy']:.2f}")
    # print(f"Fairness Score: {df['Fairness Score']:.2f}")

if __name__ == "__main__":
    main()