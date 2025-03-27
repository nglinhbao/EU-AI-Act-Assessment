#!/usr/bin/env python3
import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database connection string
db_url = os.getenv("DATA_BASE_URL")

def verify_data():
    """Query the database to verify data was imported correctly"""
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()
    
    try:
        # Get record count
        cursor.execute("SELECT COUNT(*) FROM repositories")
        count = cursor.fetchone()[0]
        print(f"Total records in database: {count}")
        
        # Get a few sample records
        cursor.execute("SELECT id, name, description, url FROM repositories LIMIT 5")
        rows = cursor.fetchall()
        
        print("\nSample records:")
        for row in rows:
            print(f"ID: {row[0]}, Name: {row[1]}")
            print(f"Description: {row[2][:100]}..." if len(row[2]) > 100 else f"Description: {row[2]}")
            print(f"URL: {row[3]}")
            print("-" * 50)
        
    except Exception as e:
        print(f"Error querying database: {e}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    verify_data() 