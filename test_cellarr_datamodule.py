#!/usr/bin/env python3

from cellarr.ml.dataloader import DataModule

# Initialize the DataModule with the dataset path
datamodule = DataModule(
    dataset_path="/braid/cellm/tiledb/cellarr_sample_splits_2025Q1/",
    cell_metadata_uri="cell_metadata",
    gene_annotation_uri="gene_annotation",
    matrix_uri="counts",  # Fixed path - it's "counts" not "assays/counts"
    label_column_name="cellTypeOntologyID",  # Using available column for labels
    study_column_name="datasetID",  # Using datasetID as the study column
    batch_size=1000,
    lognorm=True,
    target_sum=1e4,
)

print("DataModule initialized successfully!")
print(f"Dataset path: {datamodule.dataset_path}")

# Try to get some basic info about the dataset
try:
    # Setup the datamodule (this typically loads metadata and prepares data loaders)
    datamodule.setup()
    print("DataModule setup completed!")

    # Try to get train dataloader
    train_loader = datamodule.train_dataloader()
    print(f"Train dataloader created with {len(train_loader)} batches")

    # Get a single batch to check the data shape
    for batch in train_loader:
        print(f"Batch shape: {batch[0].shape if hasattr(batch[0], 'shape') else 'N/A'}")
        print(f"Batch type: {type(batch)}")
        if isinstance(batch, (list, tuple)) and len(batch) > 0:
            print(f"First element type: {type(batch[0])}")
        break

except Exception as e:
    print(f"Error during setup or data loading: {e}")
    import traceback

    traceback.print_exc()
