import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def get_system_description(txt_file_path):
    print(f"[DEBUG] Attempting to read system description from: {txt_file_path}")
    if not os.path.isfile(txt_file_path):
        print("[DEBUG] File not found. Please provide a valid .txt file path.")
        return None

    try:
        with open(txt_file_path, 'r') as file:
            system_description = file.read()
        print("[DEBUG] Successfully read system description")
        return system_description
    except Exception as e:
        print(f"[DEBUG] Error reading the .txt file: {e}")
        return None

def retrive_information_from_csv(file_name):
    print(f"[DEBUG] Attempting to read CSV file: {file_name}")
    try:
        df = pd.read_csv(file_name)
        print(f"[DEBUG] Successfully loaded CSV with {len(df)} rows")
        return df
    except Exception as e:
        print(f"[DEBUG] Error reading the CSV file: {e}")
        return None

def get_vector_db(risk_type, db_dir='./vector_db'):
    """
    Load the vector database for a specific risk type with safe deserialization.
    """
    print(f"\n[DEBUG] Loading vector DB for risk type: {risk_type}")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    safe_name = risk_type.lower().replace(' ', '_')
    db_path = os.path.join(db_dir, safe_name)
    print(f"[DEBUG] Looking for vector DB at path: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"[DEBUG] Vector DB not found at: {db_path}")
        raise ValueError(f"No vector database found for {risk_type}")
    
    print(f"[DEBUG] Successfully loaded vector DB for: {risk_type}")
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

def get_relevant_prompts(system_description, risk_type, k=3):
    """
    Retrieve k most relevant prompts from the vector database for a given risk type.
    """
    print(f"\n[DEBUG] Getting {k} relevant prompts for risk type: {risk_type}")
    try:
        db = get_vector_db(risk_type)
        print(f"[DEBUG] Performing similarity search for {risk_type}")
        results = db.similarity_search_with_score(system_description, k=k)
        prompts = [doc.page_content for doc, score in results]
        print(f"[DEBUG] Retrieved {len(prompts)} prompts for {risk_type}")
        for i, (doc, score) in enumerate(results):
            print(f"[DEBUG] Prompt {i+1} similarity score: {score}")
        return prompts
    except Exception as e:
        print(f"[DEBUG] Error retrieving prompts for {risk_type}: {e}")
        return []

def query_llama(model, tokenizer, system_description, prompt):
    print("\n[DEBUG] Querying LLaMA model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEBUG] Using device: {device}")
    
    combined_input = f"System Description: {system_description}\n\nPrompt: {prompt}\n\nAnswer: "
    print("[DEBUG] Tokenizing input")
    inputs = tokenizer(combined_input, return_tensors="pt").to(device)
    
    print("[DEBUG] Generating response")
    outputs = model.generate(**inputs, max_new_tokens=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"[DEBUG] Raw model response: {response}")

    answer = response.split("Answer: ")[-1].strip()
    print(f"[DEBUG] Extracted answer: {answer}")
    
    if answer.isdigit() and 1 <= int(answer) <= 5:
        print(f"[DEBUG] Valid numerical response: {answer}")
        return int(answer)
    else:
        print("[DEBUG] Invalid response, defaulting to 3")
        return 3

def perform_classification(row, model, tokenizer, viz=True):
    print("\n[DEBUG] Starting classification for new row")
    # Create detailed input description
    input_description = f"{row['AI System Description']}. It uses {row['Input Data Type']}. The system functions include: {row['System Functions']}. Benefits of Commercial Use: {row['Benefits of Commercial Use']}. Assumptions/Consents Regarding Data Usage: {row['Assumptions/Consents Regarding Data Usage']}"
    print("[DEBUG] Created input description")
    
    base_prompt = """
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

    risk_types = ["Unacceptable Risk", "High Risk", "Limited Risk", "Minimal Risk"]
    print(f"[DEBUG] Evaluating risk types in order: {risk_types}")
    
    for risk_type in risk_types:
        print(f"\n[DEBUG] Evaluating risk type: {risk_type}")
        prompts = get_relevant_prompts(input_description, risk_type, k=3)
        
        if not prompts:
            print(f"[DEBUG] No prompts found for {risk_type}, skipping")
            continue
            
        level_sum = 0
        prompt_count = 0
        
        for prompt_text in prompts:
            full_prompt = base_prompt + prompt_text
            print(f"\n[DEBUG] Processing prompt {prompt_count + 1} for {risk_type}")
            answer = query_llama(model, tokenizer, input_description, full_prompt)
            if viz:
                print(f"Risk Type: {risk_type}")
                print(f"Prompt: {prompt_text}")
                print(f"Response: {answer}\n")
                
            level_sum += answer
            prompt_count += 1
        
        if prompt_count > 0:
            average_level = level_sum / prompt_count
            print(f"[DEBUG] Average level for {risk_type}: {average_level}")
            if average_level > 3:
                print(f"[DEBUG] Classification decision: {risk_type} (avg_level > 3)")
                return risk_type
            
        if risk_type == "Minimal Risk":
            print("[DEBUG] Reached Minimal Risk evaluation, breaking loop")
            break
    
    print("[DEBUG] No higher risk levels met criteria, returning Minimal Risk")
    return "Minimal Risk"

def main():
    print("[DEBUG] Starting main function")
    csv_file_path = './ai_risk_prompts.csv'
    model_name = "meta-llama/Llama-2-7b-hf"

    print(f"[DEBUG] Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    print("[DEBUG] Model and tokenizer loaded successfully")
    
    synthetic_dataset = './sample.csv'
    df = retrive_information_from_csv(synthetic_dataset)
    
    print("\n[DEBUG] Starting classification for all rows")
    df['Result'] = df.apply(lambda row: perform_classification(row, model, tokenizer), axis=1)
    
    num_tp = (df['Result'] == df['System Type']).sum()
    accuracy = num_tp / len(df)

    print(f"\n[DEBUG] Final Results:")
    print(f"Total samples: {len(df)}")
    print(f"Correct predictions: {num_tp}")
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()