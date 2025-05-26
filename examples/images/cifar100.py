import os
from transformers import AutoImageProcessor, Dinov2Model
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import datamapplot

import torchdr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_features(dataset):
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
    torch.save(all_embeddings.cpu(), os.path.join(BASE_DIR, "cifar100_embeddings.pt"))

    return all_embeddings


if __name__ == "__main__":
    dataset = load_dataset("cifar100")

    embeddings_file = os.path.join(BASE_DIR, "cifar100_embeddings.pt")
    if os.path.exists(embeddings_file):
        embeddings = torch.load(embeddings_file)
    else:
        embeddings = load_features(dataset)

    # Plot the embeddings
    train_labels = dataset["train"]["fine_label"]
    test_labels = dataset["test"]["fine_label"]
    labels = train_labels + test_labels

    label_dict = {
        0: "apple",
        1: "aquarium_fish",
        2: "baby",
        3: "bear",
        4: "beaver",
        5: "bed",
        6: "bee",
        7: "beetle",
        8: "bicycle",
        9: "bottle",
        10: "bowl",
        11: "boy",
        12: "bridge",
        13: "bus",
        14: "butterfly",
        15: "camel",
        16: "can",
        17: "castle",
        18: "caterpillar",
        19: "cattle",
        20: "chair",
        21: "chimpanzee",
        22: "clock",
        23: "cloud",
        24: "cockroach",
        25: "couch",
        26: "cra",
        27: "crocodile",
        28: "cup",
        29: "dinosaur",
        30: "dolphin",
        31: "elephant",
        32: "flatfish",
        33: "forest",
        34: "fox",
        35: "girl",
        36: "hamster",
        37: "house",
        38: "kangaroo",
        39: "keyboard",
        40: "lamp",
        41: "lawn_mower",
        42: "leopard",
        43: "lion",
        44: "lizard",
        45: "lobster",
        46: "man",
        47: "maple_tree",
        48: "motorcycle",
        49: "mountain",
        50: "mouse",
        51: "mushroom",
        52: "oak_tree",
        53: "orange",
        54: "orchid",
        55: "otter",
        56: "palm_tree",
        57: "pear",
        58: "pickup_truck",
        59: "pine_tree",
        60: "plain",
        61: "plate",
        62: "poppy",
        63: "porcupine",
        64: "possum",
        65: "rabbit",
        66: "raccoon",
        67: "ray",
        68: "road",
        69: "rocket",
        70: "rose",
        71: "sea",
        72: "seal",
        73: "shark",
        74: "shrew",
        75: "skunk",
        76: "skyscraper",
        77: "snail",
        78: "snake",
        79: "spider",
        80: "squirrel",
        81: "streetcar",
        82: "sunflower",
        83: "sweet_pepper",
        84: "table",
        85: "tank",
        86: "telephone",
        87: "television",
        88: "tiger",
        89: "tractor",
        90: "train",
        91: "trout",
        92: "tulip",
        93: "turtle",
        94: "wardrobe",
        95: "whale",
        96: "willow_tree",
        97: "wolf",
        98: "woman",
        99: "worm",
    }
    vectorized_converter = np.vectorize(lambda x: label_dict[x])
    labels_str = vectorized_converter(labels)

    # Dimensionality reduction
    z = torchdr.TSNE(
        n_components=2,
        verbose=True,
        perplexity=50,
        device=device,
        max_iter=2000,
        backend="faiss",
    ).fit_transform(embeddings)
    z = z.cpu().numpy()

    fig, ax = datamapplot.create_plot(
        z, labels_str, label_over_points=True, label_font_size=30
    )
    fig.savefig("datamapplot_cifar100.png", bbox_inches="tight")
