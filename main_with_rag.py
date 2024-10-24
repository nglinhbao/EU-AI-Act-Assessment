import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os

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

def get_prompts_from_db(system_description: str, db_paths: dict, embeddings, k: int = 3):
    """
    Retrieve relevant prompts from vector databases for each risk type.
    
    Args:
        system_description (str): Description of the system to classify
        db_paths (dict): Dictionary mapping risk types to database paths
        embeddings: HuggingFace embeddings instance
        k (int): Number of prompts to retrieve per risk type
    
    Returns:
        list: List of tuples containing (risk_type, prompt)
    """
    prompts = []
    
    for risk_type, db_path in db_paths.items():
        # Load the vector database for this risk type
        db = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
        
        # Get similar documents
        results = db.similarity_search_with_score(system_description, k=k)
        
        # Extract prompts and add to list with risk type
        for doc, score in results:
            prompts.append({
                'Type': risk_type,
                'Prompt': doc.page_content  # Assuming prompt is stored in page_content
            })
    
    # Convert to DataFrame to maintain compatibility with existing code
    return pd.DataFrame(prompts)

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

def perform_classification(row, model, tokenizer, vector_db_prompts, viz=True):
    # Create a more detailed input description using multiple columns
    input_description = f"{row['AI System Description']}. It uses {row['Input Data Type']}. The system functions include: {row['System Functions']}. Benefits of Commercial Use: {row['Benefits of Commercial Use']}. Assumptions/Consents Regarding Data Usage: {row['Assumptions/Consents Regarding Data Usage']}"
    
    # Base prompt template
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
    
    for _, prompt_row in vector_db_prompts.iterrows():
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
    model_name = "meta-llama/Llama-2-7b-hf"

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    # Initialize embeddings for vector DB
    model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Define paths to vector databases for each risk type
    db_paths = {
        "Unacceptable Risk": "./data/unacceptable_risk_db",
        "High Risk": "./data/high_risk_db",
        "Limited Risk": "./data/limited_risk_db",
        "Minimal Risk": "./data/minimal_risk_db"
    }
    
    # Load the dataset to classify
    synthetic_dataset = './sample.csv'
    df = retrive_information_from_csv(synthetic_dataset)
    
    if df is not None:
        # Process each row
        for index, row in df.iterrows():
            # Get system description for vector DB query
            system_desc = (f"{row['AI System Description']} {row['Input Data Type']} "
                         f"{row['System Functions']} {row['Benefits of Commercial Use']} "
                         f"{row['Assumptions/Consents Regarding Data Usage']}")
            
            # Get relevant prompts from vector DB
            vector_db_prompts = get_prompts_from_db(system_desc, db_paths, embeddings)
            
            # Perform classification using retrieved prompts
            result = perform_classification(row, model, tokenizer, vector_db_prompts)
            df.at[index, 'Result'] = result
        
        # Compute accuracy
        num_tp = (df['Result'] == df['System Type']).sum()
        accuracy = num_tp / len(df)
        print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()