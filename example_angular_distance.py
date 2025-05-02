import time

import torch
from scipy.spatial.distance import cdist

from torchdr.utils.geometry import pairwise_distances

# Set device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate random matrices X and Y with 30000 rows and 128 dimensions
X = torch.randn(30000, 128, device=device)
Y = torch.randn(30000, 128, device=device)

# Start timing
start_time = time.time()

# Compute the angular pairwise distance matrix
C, _ = pairwise_distances(X, Y, metric="angular", backend=None)

print(C)

# End timing
end_time = time.time()

# # Print the shape of the resulting distance matrix
# print('Shape of the angular pairwise distance matrix:', C.shape)

# Print the time taken for the computation
print("Time taken for computation:", end_time - start_time, "seconds")

# Compute the angular pairwise distance matrix using SciPy
start_time_scipy = time.time()
C_scipy = cdist(X.cpu().numpy(), Y.cpu().numpy(), metric="cosine")
end_time_scipy = time.time()

# Print the time taken for the SciPy computation
print("Time taken for SciPy computation:", end_time_scipy - start_time_scipy, "seconds")
