import numpy as np


print("Loading processed data...")
X = np.load('X_processed.npy')
y = np.load('y_processed.npy')
print(f"Features (X) shape: {X.shape}")
print(f"Labels (y) shape: {y.shape}")
print("\nFirst 5 rows of features (X):")
print(X[:5])
print("\nFirst 10 labels (y):")
print(y[:10])
print(f"\nTarget distribution:")
print(f"Normal traffic (0): {sum(y == 0)}")
print(f"Attack traffic (1): {sum(y == 1)}")
print(f"\nFeature statistics:")
print(f"Min: {X[:, 0].min()}")
print(f"Max: {X[:, 0].max()}")
print(f"Mean: {X[:, 0].mean():.3f}")