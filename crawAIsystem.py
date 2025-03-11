import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()

# GitHub API URL
base_url = "https://api.github.com/search/repositories"

# Your search query
query = "AI"

# Number of pages to fetch
num_pages = 20

# Your GitHub personal access token
access_token = os.getenv("GITHUB_ACCESS_TOKEN")

# Headers for authentication
headers = {
    "Authorization": f"token {access_token}"
}

# Function to fetch repositories
def fetch_repositories(page):
    params = {
        "q": query,
        "per_page": 30,  # Number of results per page
        "page": page
    }
    response = requests.get(base_url, headers=headers, params=params)
    return response.json()

# List to store repository data
repositories = []

# Fetch data from each page
for page in range(1, num_pages + 1):
    data = fetch_repositories(page)
    for item in data.get('items', []):
        repo_info = {
            "name": item['name'],
            "description": item['description'],
            "url": item['html_url']
        }
        repositories.append(repo_info)

# Save data to a JSON file
with open('repositories.json', 'w') as f:
    json.dump(repositories, f, indent=4)

print("Data fetched and saved to repositories.json")
