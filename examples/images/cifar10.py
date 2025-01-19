from transformers import AutoImageProcessor, Dinov2Model
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load CIFAR-10 dataset
dataset = load_dataset("cifar10")
test_images = dataset["train"]["img"]

# Load the image processor and model
image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(
    device
)  # Move model to GPU
model.eval()


# Preprocess images into tensors
def preprocess(images):
    # Process all images in the batch
    inputs = image_processor(images, return_tensors="pt")
    return {
        key: tensor.to(device) for key, tensor in inputs.items()
    }  # Move tensors to GPU


# DataLoader for batching
batch_size = 1024
test_dataloader = DataLoader(
    test_images, batch_size=batch_size, collate_fn=lambda batch: preprocess(batch)
)

# Embed all images
embeddings = []
with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Processing Images"):
        outputs = model(**batch)
        embeddings.append(
            outputs.last_hidden_state.mean(dim=1)
        )  # Take mean of spatial dimensions for each image

# Concatenate all embeddings
all_embeddings = torch.cat(embeddings, dim=0)

# Print shape of embeddings
print("Shape of all embeddings:", all_embeddings.shape)


from sklearn.preprocessing import LabelEncoder
import torchdr
import umap
import matplotlib.pyplot as plt

# Label encode the labels
labels = dataset["test"]["label"]
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# z_ = torchdr.PCA(n_components=50, device="cuda").fit_transform(all_embeddings)

z_cpu = all_embeddings.cpu().numpy()
z1 = umap.UMAP(n_components=2, verbose=True, n_neighbors=50).fit_transform(z_cpu)

# Dimensionality reduction
z2 = torchdr.UMAP(
    n_components=2, verbose=True, n_neighbors=50, device="cuda", sparsity=False
).fit_transform(all_embeddings)
z2 = z2.cpu().numpy()


# Plot the results for both UMAP projections
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# UMAP without PCA
scatter1 = axes[0].scatter(
    z1[:, 0], z1[:, 1], c=encoded_labels, cmap="tab10", alpha=0.7
)
axes[0].set_title("UMAP on Original Embeddings")
axes[0].set_xlabel("UMAP Component 1")
axes[0].set_ylabel("UMAP Component 2")
fig.colorbar(
    scatter1, ax=axes[0], ticks=range(len(label_encoder.classes_)), label="Labels"
)

# UMAP after PCA
scatter2 = axes[1].scatter(
    z2[:, 0], z2[:, 1], c=encoded_labels, cmap="tab10", alpha=0.7
)
axes[1].set_title("UMAP on PCA-reduced Embeddings")
axes[1].set_xlabel("UMAP Component 1")
axes[1].set_ylabel("UMAP Component 2")
fig.colorbar(
    scatter2, ax=axes[1], ticks=range(len(label_encoder.classes_)), label="Labels"
)

# Display the plot
plt.tight_layout()
plt.savefig("cifar10_umap_comparison.png")
plt.show()
