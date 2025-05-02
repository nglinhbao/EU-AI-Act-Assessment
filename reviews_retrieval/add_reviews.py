import pandas as pd
from google_play_scraper import reviews, Sort

def get_app_reviews(app_id, app_name, max_reviews=1000):
    """Retrieve reviews for a specific app"""
    print(f"Fetching reviews for {app_name} ({app_id})...")
    try:
        # Use reviews() instead of reviews_all() to limit the number of reviews
        review_results, continuation_token = reviews(
            app_id,
            lang="en",
            country="us",
            sort=Sort.NEWEST,
            count=max_reviews  # This limits the number of reviews
        )

        # Print the structure of the first review for debugging
        if review_results:
            print(f"Review keys available: {list(review_results[0].keys())}")

        # Convert to a structured format with safer field access
        review_data = []
        for review in review_results:
            review_item = {
                'content': review.get('content', ''),  # Review content
                'score': review.get('score', None),    # Review score (rating)
                'at': review.get('at', None),          # Review timestamp
                'app_id': app_id,                      # App ID
                'app_name': app_name                   # App Name
            }
            review_data.append(review_item)

        return review_data
    except Exception as e:
        print(f"Error retrieving reviews for {app_name}: {str(e)}")
        # Print a full traceback for debugging
        import traceback
        traceback.print_exc()
        return []

def create_reviews_dataset(apps, output_path, max_reviews=1000):
    """Fetch reviews for each app and save them to a new dataset."""
    all_reviews = []  # List to store all reviews

    for app in apps:
        app_id = app['Link'].split("id=")[-1]
        app_name = app['App Name']
        try:
            # Fetch reviews for the app
            review_data = get_app_reviews(app_id, app_name, max_reviews)
            # Append the reviews to the all_reviews list
            all_reviews.extend(review_data)
        except Exception as e:
            print(f"Error fetching reviews for {app_name}: {e}")

    # Convert the reviews list to a DataFrame
    reviews_df = pd.DataFrame(all_reviews)

    # Save the reviews DataFrame to the specified output path
    reviews_df.to_csv(output_path, index=False)
    print(f"Reviews dataset created successfully at {output_path}!")

# Load the dataset
dataset_path = "./datasets/full_dataset/AI_apps_full_dataset.csv"
apps_df = pd.read_csv(dataset_path)

# Convert the dataframe to a list of dictionaries for processing
apps = apps_df.to_dict(orient="records")

# Define the output path for the reviews dataset
output_reviews_path = "./datasets/full_dataset/corresponding_reviews.csv"

# Fetch reviews and create the new dataset
create_reviews_dataset(apps, output_reviews_path)

print("Reviews dataset created successfully!")