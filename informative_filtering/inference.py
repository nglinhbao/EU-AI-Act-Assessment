import pandas as pd
import numpy as np
from ar_miner import ARMiner

# Set up datetime for logging
from datetime import datetime
print(f"Running inference at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")

# Load the trained model
model = ARMiner.load_model('models/ar_miner_model.pkl')

# Load the inference data
inference_data = pd.read_csv('inference_dataset.csv')

# Filter for informative reviews using the modified function
is_informative, probabilities = model.filter_reviews(inference_data['content'])

# Add results to the original dataframe
inference_data['informative'] = is_informative
inference_data['informative_prob'] = probabilities

# Extract informative reviews
informative_reviews = inference_data[inference_data['informative']].copy()

# Save the filtered reviews
informative_reviews.to_csv('informative_reviews.csv', index=False)

# Print statistics
print(f"Total reviews: {len(inference_data)}")
print(f"Informative reviews: {len(informative_reviews)}")
print(f"Percentage: {len(informative_reviews) / len(inference_data) * 100:.2f}%")

# Print sample of informative reviews
print("\nSample of informative reviews:")
if len(informative_reviews) > 0:
    pd.set_option('display.max_colwidth', 100)
    print(informative_reviews[['content', 'informative_prob']].head())