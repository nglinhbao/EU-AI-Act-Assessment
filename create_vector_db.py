from langchain.document_loaders import DataFrameLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import pandas as pd
from typing import List, Dict
import os

class RiskDatabaseSystem:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b"):
        """
        Initialize the risk database system with Llama 2 embeddings.
        
        Args:
            model_name (str): Name of the Hugging Face model to use for embeddings
        """
        # Configure Llama 2 embeddings
        model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        self.databases = {}
        self.risk_categories = [
            "Unacceptable Risk",
            "High Risk",
            "Limited Risk",
            "Minimal Risk"
        ]
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better embedding quality.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Limit text length to prevent token overflow
        max_chars = 512  # Adjust based on model's context window
        if len(text) > max_chars:
            text = text[:max_chars] + '...'
        return text
        
    def process_data(self, data: List[Dict]) -> Dict[str, pd.DataFrame]:
        """
        Process the input data and organize it by risk category.
        
        Args:
            data (List[Dict]): List of dictionaries containing AI system data
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of DataFrames organized by risk category
        """
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create separate DataFrames for each risk category
        categorized_data = {}
        for risk_category in self.risk_categories:
            category_df = df[df['System Type'] == risk_category].copy()
            if not category_df.empty:
                # Preprocess text columns
                text_columns = ['AI System Description', 'System Functions', 
                              'Benefits of Commercial Use', 'Assumptions/Consents Regarding Data Usage']
                for col in text_columns:
                    category_df[col] = category_df[col].apply(self.preprocess_text)
                categorized_data[risk_category] = category_df
                
        return categorized_data
    
    def create_vector_databases(self, categorized_data: Dict[str, pd.DataFrame], batch_size: int = 32):
        """
        Create vector databases for each risk category using Chroma.
        
        Args:
            categorized_data (Dict[str, pd.DataFrame]): Dictionary of DataFrames by risk category
            batch_size (int): Batch size for processing documents
        """
        for risk_category, df in categorized_data.items():
            # Create documents from DataFrame
            documents = []
            for _, row in df.iterrows():
                content = (
                    f"System Description: {row['AI System Description']}\n"
                    f"Input Data Type: {row['Input Data Type']}\n"
                    f"System Functions: {row['System Functions']}\n"
                    f"Benefits: {row['Benefits of Commercial Use']}\n"
                    f"Assumptions: {row['Assumptions/Consents Regarding Data Usage']}"
                )
                metadata = {
                    "risk_category": risk_category,
                    "system_description": row['AI System Description'],
                    "data_type": row['Input Data Type']
                }
                documents.append(Document(page_content=content, metadata=metadata))
            
            # Process documents in batches
            db_name = f"{risk_category.lower().replace(' ', '_')}_db"
            persist_directory = f"./data/{db_name}"
            
            # Create vector database with batching
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                if i == 0:
                    self.databases[risk_category] = Chroma.from_documents(
                        documents=batch,
                        embedding=self.embeddings,
                        persist_directory=persist_directory
                    )
                else:
                    self.databases[risk_category].add_documents(batch)
            
            # Persist the database
            self.databases[risk_category].persist()
    
    def search_systems(self, query: str, risk_category: str = None, k: int = 5):
        """
        Search for similar systems across all or specific risk categories.
        
        Args:
            query (str): Search query
            risk_category (str, optional): Specific risk category to search
            k (int): Number of results to return
            
        Returns:
            List of similar documents with their scores
        """
        # Preprocess query
        query = self.preprocess_text(query)
        
        if risk_category and risk_category in self.databases:
            return self.databases[risk_category].similarity_search_with_score(query, k=k)
        
        # Search across all databases if no specific category is provided
        results = []
        for category, db in self.databases.items():
            category_results = db.similarity_search_with_score(query, k=k)
            results.extend(category_results)
        
        # Sort by similarity score and return top k results
        results.sort(key=lambda x: x[1])
        return results[:k]

    def analyze_results(self, results: List[tuple]) -> Dict:
        """
        Analyze search results to provide insights.
        
        Args:
            results: List of (document, score) tuples from search
            
        Returns:
            Dict: Analysis of the results
        """
        analysis = {
            "risk_distribution": {},
            "avg_similarity_score": 0,
            "common_data_types": {},
            "top_score": 0,
            "lowest_score": float('inf')
        }
        
        total_score = 0
        for doc, score in results:
            # Update risk distribution
            risk_cat = doc.metadata["risk_category"]
            analysis["risk_distribution"][risk_cat] = analysis["risk_distribution"].get(risk_cat, 0) + 1
            
            # Update data types
            data_type = doc.metadata["data_type"]
            analysis["common_data_types"][data_type] = analysis["common_data_types"].get(data_type, 0) + 1
            
            # Update scores
            total_score += score
            analysis["top_score"] = max(analysis["top_score"], score)
            analysis["lowest_score"] = min(analysis["lowest_score"], score)
        
        analysis["avg_similarity_score"] = total_score / len(results) if results else 0
        return analysis

    def get_system_statistics(self) -> Dict:
        """
        Get statistics about the systems in each risk category.
        
        Returns:
            Dict: Statistics about the systems
        """
        stats = {}
        for category, db in self.databases.items():
            collection = db.get()
            data_types = [doc.metadata["data_type"] for doc in collection["documents"]]
            
            stats[category] = {
                "total_systems": len(collection["ids"]),
                "unique_data_types": len(set(data_types)),
                "data_type_distribution": pd.Series(data_types).value_counts().to_dict()
            }
        return stats

def main():
    """
    Example usage of the RiskDatabaseSystem
    """
    # Initialize system with Llama 2 embeddings
    system = RiskDatabaseSystem()
    
    # Sample data processing
    sample_data = [
        {
            "System Type": "High Risk",
            "AI System Description": "Credit scoring system",
            "Input Data Type": "Financial data",
            "System Functions": "Evaluates creditworthiness",
            "Benefits of Commercial Use": "Faster loan decisions",
            "Assumptions/Consents Regarding Data Usage": "Explicit user consent"
        }
        # Add more sample data as needed
    ]
    
    # Process and create databases
    categorized_data = system.process_data(sample_data)
    system.create_vector_databases(categorized_data)
    
    # Example search
    results = system.search_systems("credit evaluation system")
    
    # Analyze results
    analysis = system.analyze_results(results)
    print("Search Analysis:", analysis)
    
    # Get system statistics
    stats = system.get_system_statistics()
    print("System Statistics:", stats)

if __name__ == "__main__":
    main()