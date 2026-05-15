from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib

# Dummy training data
X = np.random.rand(100, 6)
y = np.random.randint(0, 2, 100)

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save clean model
joblib.dump(model, "model.pkl")

print("New model.pkl created successfully")