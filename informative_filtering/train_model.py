import os
from ar_miner import ARMiner

# Define the app names from your dataset
app_names = ['facebook', 'swiftkey', 'tapfish', 'templerun2']

# Create and train the AR-Miner model
model = ARMiner()
model.train(app_names)

# Save the model
os.makedirs('models', exist_ok=True)
model.save_model('models/ar_miner_model.pkl')