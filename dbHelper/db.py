# db.py
import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database connection string
db_url = os.getenv("DATA_BASE_URL")

def init_db():
    """Initialize the database with required tables"""
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()
    
    try:
        # Create repositories table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS repositories (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            url VARCHAR(512) NOT NULL,
            cleaned_desc TEXT
        )
        """)
        
        conn.commit()
        print("Database initialized successfully")
    except Exception as e:
        conn.rollback()
        print(f"Error initializing database: {e}")
    finally:
        cursor.close()
        conn.close()

def import_data():
    """Import data from CSV to the database"""
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()
    
    try:
        # Clear existing data if needed
        cursor.execute("TRUNCATE TABLE repositories")
        
        # Read CSV file
        df = pd.read_csv('datasets/repositories_cleaned.csv')
        print(f"Parsed {len(df)} records from CSV")
        
        # Insert data in batches
        batch_size = 100
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            
            # Prepare values for insertion
            values = []
            for _, row in batch.iterrows():
                # Handle NULL values and escape single quotes
                name = row['name'].replace("'", "''") if pd.notna(row['name']) else ''
                description = row['description'].replace("'", "''") if pd.notna(row['description']) else ''
                url = row['url'].replace("'", "''") if pd.notna(row['url']) else ''
                cleaned_desc = row['cleaned_desc'].replace("'", "''") if pd.notna(row['cleaned_desc']) else ''
                
                values.append(f"('{name}', '{description}', '{url}', '{cleaned_desc}')")
            
            if values:
                query = f"""
                INSERT INTO repositories (name, description, url, cleaned_desc)
                VALUES {', '.join(values)}
                """
                cursor.execute(query)
        
        conn.commit()
        print("Data import completed successfully")
    except Exception as e:
        conn.rollback()
        print(f"Error importing data: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    # Run initialization and data import
    init_db()
    import_data()