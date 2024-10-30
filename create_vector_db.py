import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def create_vector_databases():
    # Read the CSV file
    df = pd.read_csv('datasets/ai_risk_prompts.csv')
    
    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create output directory if it doesn't exist
    os.makedirs('./vector_db', exist_ok=True)
    
    # Group the prompts by risk type and create vector databases
    for risk_type, group in df.groupby('Type'):
        # Collect prompts for this risk type
        texts = group['Prompt'].dropna().tolist()
        
        # Only create a vector database if we have prompts
        if texts:
            print(texts)
            # Create the vector store for this risk type
            db = FAISS.from_texts(
                texts,
                embeddings,
                metadatas=[{"risk_type": risk_type} for _ in texts]
            )
            
            # Save to disk with risk type as the folder name
            db_path = f"./vector_db/{risk_type.lower().replace(' ', '_')}"
            db.save_local(db_path)
            print(f"Created vector database for {risk_type} at {db_path}")

if __name__ == "__main__":
    create_vector_databases()
