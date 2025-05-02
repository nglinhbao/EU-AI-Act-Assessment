import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import re
# from fairness_scoring import evaluate_app_fairness, load_models

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

def get_review_vector_db(app_name, db_dir='./vector_db'):
    """
    Load the vector database for a specific app with safe deserialization.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cuda'}
    )
    
    safe_name = app_name.lower().replace(' ', '_').replace(':', '_')
    db_path = os.path.join(db_dir, safe_name)
    
    if not os.path.exists(db_path):
        print(f"No vector database found for {app_name}, trying default database")
        # Try to load a default database if app-specific one doesn't exist
        db_path = os.path.join(db_dir, "polybuzz_formerly_poly_ai")
        if not os.path.exists(db_path):
            return None
    
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

def get_relevant_reviews(input_description, app_name, k=5):
    """
    Retrieve k most relevant reviews from the vector database for the given app
    based on similarity to the input description.
    """
    try:
        db = get_review_vector_db(app_name)
        if db is None:
            return []
            
        results = db.similarity_search_with_score(input_description, k=k)
        
        # Extract reviews and their metadata
        reviews = []
        for doc, score in results:
            review_text = doc.page_content
            metadata = doc.metadata
            reviews.append({
                "content": review_text,
                "score": metadata.get("score", "Unknown"),
                "informative_prob": metadata.get("informative_prob", 0)
            })
            
        return reviews
    except Exception as e:
        print(f"Error retrieving reviews for {app_name}: {e}")
        return []

def format_reviews(reviews):
    """Format a list of reviews into a readable string."""
    if not reviews:
        return "No relevant user reviews found."
        
    formatted = "Relevant user reviews:\n"
    for i, review in enumerate(reviews, 1):
        rating = review.get("score", "Unknown")
        formatted += f"{i}. Rating: {rating}/5 - {review['content']}\n"
    
    return formatted

def query_llama(model, tokenizer, system_description, prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    combined_input = f"System Description: {system_description}\n\nPrompt: {prompt}"
    inputs = tokenizer(combined_input, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:        
        # Extract the second "Answer: "
        answer_parts = response.split("Answer: ")
        print(answer_parts)
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
    import torch

    difference_analysis = row['Difference Analysis']

    base_prompt = """
        Are Promised Features and User Reviews inconsistent?

        Answer the question only with 'Yes' or 'No'

        Template:
        Answer:
        """

    full_prompt = difference_analysis + base_prompt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(response)
    # Extract answer after "Answer:"
    if "Answer:" in response:
        answer = response.split("Answer:")[-1].strip().split()[0]
        print(answer)
        return answer.lower() == "yes"
    return False

def perform_classification(row, model, tokenizer, prompts_df, viz=True):
    # Base description without reviews
    base_description = (
        f"{row['Full Description']}. "
        f"Additional app information: {row['App Info Modal']}. "
        f"Data shared with third parties: {row['Shared Data']}. "
        f"Data collected by the app: {row['Collected Data']}. "
        f"Security practices: {row['Security Practices']}."
    )
    
    # Retrieve relevant reviews from vector database
    app_name = row["App Name"]
    reviews = get_relevant_reviews(base_description, app_name, k=8)
    
    # Format reviews and append to input description
    input_description = base_description
    if reviews:
        formatted_reviews = format_reviews(reviews)
        input_description += f"\n\n{formatted_reviews}"
    
    # # Include additional review analysis if available
    # if check_inconsistency(row, model, tokenizer):
    #     if evaluate_app_fairness(app_name, word2vec_model, word2vec_words, classifier_model, use_model) >= 0.1:
    #         input_description += f"\nOfficial user review analysis: {row['User Review Analysis']}."
    
    base_prompt = """
        Answer the question with 'Yes' or 'No'
        (only answer the main question and do not split the question)
        
        Template:
        Answer: [Yes/No]
        Reasoning: [reasons why you give the score in 1 short paragraph]
    """

    # Initialize tracking variables
    # Removed "Minimal Risk" as it's no longer in the prompts
    risk_types = ["Unacceptable risk", "High risk", "Limited risk"]
    current_reasoning = ""
    
    # Process each risk type
    for risk_type in risk_types:
        # Filter prompts by risk type
        risk_prompts = prompts_df[prompts_df['Type'] == risk_type]
        
        if risk_prompts.empty:  # If no prompts for this risk type, skip
            continue
            
        score_sum = 0
        prompt_count = 0
        
        # Process each prompt for this risk type
        for _, prompt_row in risk_prompts.iterrows():
            full_prompt = base_prompt + prompt_row['Prompt']
            response = query_llama(model, tokenizer, input_description, full_prompt)
            
            if viz:
                print(f"Risk Type: {risk_type}")
                print(f"Score: {response['score']}")
                print(f"Reasoning: {response['reasoning']}\n")
                
            score_sum += response['score']
            prompt_count += 1
            current_reasoning += response['reasoning']
        
            # Check if we have enough evidence to classify
            if prompt_count > 0:
                confidence_score = score_sum / prompt_count
                if response['score'] == 1:
                    return {
                        'risk_type': risk_type,
                        'confidence_score': confidence_score,
                        'all_reasoning': current_reasoning,
                        'current_reasoning': response['reasoning']
                    }
    
    # If we've processed all risk types and found no classification,
    # default to minimal risk
    return {
        'risk_type': "Minimal Risk",
        'confidence_score': 0,  # Default confidence since we're using it as fallback
        'all_reasoning': current_reasoning if current_reasoning else "No specific risks identified in the higher risk categories.",
        'current_reasoning': ""
    }

def main():
    questions = './datasets/EU_AI_Act_Assessment_Questions.csv'
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    
    synthetic_dataset = './datasets/full_dataset/AI_apps_full_dataset.csv'
    df = retrive_information_from_csv(synthetic_dataset)

    # Load all prompts from CSV
    prompts_df = pd.read_csv(questions)
    
    # Prepare the output file
    output_file = './datasets/results.csv'
    with open(output_file, 'w') as f:
        # Write the header
        f.write("App Name;Predicted System Type;All Reasoning;Current Reasoning\n")

    # Process each row and save results on the fly
    for _, row in df.iterrows():
        result = perform_classification(row, model, tokenizer, prompts_df)
        with open(output_file, 'a') as f:
            f.write(f"{row['App Name']};{result['risk_type']};{result['all_reasoning']};{result['current_reasoning']}\n")

if __name__ == "__main__":
    main()