import os
from transformers import AutoImageProcessor, Dinov2Model
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import datamapplot
import matplotlib.pyplot as plt

import torchdr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_features():
    # Load dataset
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
    torch.save(all_embeddings.cpu(), "cifar10_embeddings.pt")

    return all_embeddings


if __name__ == "__main__":
    if os.path.exists("cifar10_embeddings.pt"):
        embeddings = torch.load("cifar10_embeddings.pt")
    else:
        embeddings = load_features()

    # Plot the embeddings
    dataset = load_dataset("cifar10")

    train_labels = dataset["train"]["label"]
    test_labels = dataset["test"]["label"]
    labels = train_labels + test_labels

    label_dict = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }
    vectorized_converter = np.vectorize(lambda x: label_dict[x])
    labels_str = vectorized_converter(labels)

    # Dimensionality reduction
    z = torchdr.UMAP(
        n_components=2,
        verbose=True,
        n_neighbors=50,
        device=device,
    ).fit_transform(embeddings)
    z = z.cpu().numpy()

    fig, ax = datamapplot.create_plot(
        z,
        labels_str,
        label_over_points=True,
        label_font_size=20,
    )
    fig.savefig("datamapplot_cifar10.png", bbox_inches="tight")
