import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def create_vector_databases():
    # Read the CSV file
    df = pd.read_csv('datasets/full_dataset/filtered_corresponding_reviews.csv')
    
    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create output directory if it doesn't exist
    os.makedirs('./vector_db', exist_ok=True)
    
    # Group the reviews by app_name and create vector databases
    for app_name, group in df.groupby('app_name'):
        # Collect review content for this app
        texts = group['content'].dropna().tolist()
        
        # Only create a vector database if we have content
        if texts:
            print(f"Processing {len(texts)} reviews for {app_name}")
            
            # Create metadata for each review
            metadatas = []
            for _, row in group.iterrows():
                metadata = {
                    "app_name": row['app_name'],
                    "reviewer": None,  # 'reviewer' column is not in the CSV, set to None
                    "date": row['at'],  # Use 'at' column for the date
                    "score": row['score'],
                    "informative": row['informative'],
                    "informative_prob": row['informative_prob']
                }
                metadatas.append(metadata)
            
            # Create the vector store for this app
            db = FAISS.from_texts(
                texts,
                embeddings,
                metadatas=metadatas
            )
            
            # Save to disk with app_name as the folder name (sanitized)
            safe_name = app_name.lower().replace(' ', '_').replace(':', '_')
            db_path = f"./vector_db/{safe_name}"
            db.save_local(db_path)
            print(f"Created vector database for {app_name} at {db_path}")

if __name__ == "__main__":
    create_vector_databases()