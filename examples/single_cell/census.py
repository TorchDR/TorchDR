import warnings
from typing import List

import anndata
import cellxgene_census
import numpy as np
import scanpy as sc
import datamapplot
import matplotlib.pyplot as plt

from torchdr import LargeVis


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


# human embeddings
CENSUS_VERSION = "2023-12-15"
EXPERIMENT_NAME = "homo_sapiens"

# These are embeddings available to this Census version
embedding_names = ["scgpt"]

census = cellxgene_census.open_soma(census_version=CENSUS_VERSION)


obs_value_filter = (
    "tissue_general=='lung' and dataset_id=='53d208b0-2cfd-4366-9866-c3c6114081bc'"
)

adata = cellxgene_census.get_anndata(
    census=census,
    organism=EXPERIMENT_NAME,
    obs_value_filter=obs_value_filter,
    obs_embeddings=embedding_names,
)

adata = remove_missing_embedding_cells(adata, embedding_names)


labels_dict = adata.obs.to_dict(orient="list")

embeddings = adata.obsm["scgpt"]


Z = LargeVis(device="cuda", backend="keops", verbose=True).fit_transform(embeddings)

fig, ax = datamapplot.create_plot(
    Z,
    labels_dict["cell_type"],
    label_over_points=True,
    dynamic_label_size=True,
    min_font_size=7,
    figsize=(5, 5),
)
plt.tight_layout()
plt.savefig("scgpt_cell_type.pdf", bbox_inches="tight")


fig, ax = datamapplot.create_plot(
    Z,
    labels_dict["sex"],
    label_over_points=True,
    dynamic_label_size=True,
    min_font_size=10,
    figsize=(5, 5),
)
plt.tight_layout()
plt.savefig("scgpt_sex.pdf", bbox_inches="tight")


fig, ax = datamapplot.create_plot(
    Z,
    labels_dict["development_stage"],
    label_over_points=False,
    dynamic_label_size=True,
    min_font_size=10,
    figsize=(5, 5),
)
plt.tight_layout()
plt.savefig("scgpt_dv_stage.pdf", bbox_inches="tight")
