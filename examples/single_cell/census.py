import warnings
from typing import List

import anndata
import cellxgene_census
import numpy as np
import scanpy as sc

# human embeddings
CENSUS_VERSION = "2023-12-15"
EXPERIMENT_NAME = "homo_sapiens"

# These are embeddings available to this Census version
embedding_names = ["geneformer", "scvi", "scgpt", "uce"]

census = cellxgene_census.open_soma(census_version=CENSUS_VERSION)

# Let's find our cells of interest
obs_value_filter = "tissue_general=='eye' and is_primary_data == True"

obs_df = cellxgene_census.get_obs(
    census, EXPERIMENT_NAME, value_filter=obs_value_filter, column_names=["soma_joinid"]
)
soma_joinids_subset = obs_df["soma_joinid"].values.tolist()


def remove_missing_embedding_cells(adata: anndata.AnnData, emb_names: List[str]):
    """Embeddings with missing data contain all NaN,
    so we must find the intersection of non-NaN rows in the fetched embeddings
    and subset the AnnData accordingly.
    """
    filt = np.ones(adata.shape[0], dtype="bool")
    for key in emb_names:
        nan_row_sums = np.sum(np.isnan(adata.obsm[key]), axis=1)
        total_columns = adata.obsm[key].shape[1]
        filt = filt & (nan_row_sums != total_columns)
    adata = adata[filt].copy()

    return adata


# Let's get the AnnData
adata = cellxgene_census.get_anndata(
    census=census,
    organism=EXPERIMENT_NAME,
    obs_coords=soma_joinids_subset,
    obs_embeddings=embedding_names,
)

adata = remove_missing_embedding_cells(adata, embedding_names)

print(adata)
