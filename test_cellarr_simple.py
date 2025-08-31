#!/usr/bin/env python3


from cellarr import CellArrDataset

# Try using CellArrDataset directly without the DataModule
dataset_path = "/braid/cellm/tiledb/cellarr_sample_splits_2025Q1"

cd = CellArrDataset(
    dataset_path=dataset_path,
    assay_tiledb_group="",
)
