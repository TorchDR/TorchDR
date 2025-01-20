import os
from transformers import AutoImageProcessor, Dinov2Model
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
import torchdr
import umap
import matplotlib.pyplot as plt

# import datamapplot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_features():
    # Load dataset
    dataset = load_dataset("cifar100")
    train_images = dataset["train"]["img"]
    test_images = dataset["test"]["img"]
    images = train_images + test_images

    # Load the image processor and model
    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device)
    model.eval()

    def preprocess(images):
        inputs = image_processor(images, return_tensors="pt")
        return {key: tensor.to(device) for key, tensor in inputs.items()}

    # DataLoader for batching
    batch_size = 1024
    test_dataloader = DataLoader(
        images, batch_size=batch_size, collate_fn=lambda batch: preprocess(batch)
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
    print("Shape of the embeddings:", all_embeddings.shape)
    torch.save(all_embeddings.cpu(), "cifar100_embeddings.pt")

    return all_embeddings


if __name__ == "__main__":
    if os.path.exists("cifar100_embeddings.pt"):
        embeddings = torch.load("cifar100_embeddings.pt")
    else:
        embeddings = load_features()

# # Label encode the labels
# label_encoder = LabelEncoder()
# encoded_labels = label_encoder.fit_transform(labels)

# # z_cpu = all_embeddings.cpu().numpy()
# # z1 = umap.UMAP(n_components=2, verbose=True, n_neighbors=50).fit_transform(z_cpu)

# z_ = torchdr.PCA(n_components=100, device="cuda").fit_transform(all_embeddings)
# embeddings = torchdr.UMAP(
#     n_components=2, verbose=True, n_neighbors=50, device="cuda"
# ).fit_transform(z_)
# embeddings = embeddings.cpu().numpy()

# datamapplot.create_plot(embeddings, labels)

# plt.savefig("cifar100_fig.png")
