#!/usr/bin/env python3

import tiledb

# Open the cell metadata to see what columns are available
cell_metadata_path = "/braid/cellm/tiledb/cellarr_sample_splits_2025Q1/cell_metadata"

try:
    with tiledb.open(cell_metadata_path, "r") as tdb:
        print("Cell metadata schema:")
        print(f"Number of attributes: {tdb.schema.nattr}")
        print("\nAvailable attributes (columns):")
        for i in range(tdb.schema.nattr):
            attr = tdb.schema.attr(i)
            print(f"  - {attr.name} ({attr.dtype})")

        # Try to read a small sample of data to see what's there
        print("\nSample data (first 5 rows):")
        df = tdb.df[0:5]
        print(df.columns.tolist())
        print(df.head())

except Exception as e:
    print(f"Error reading cell metadata: {e}")
    import traceback

    traceback.print_exc()
