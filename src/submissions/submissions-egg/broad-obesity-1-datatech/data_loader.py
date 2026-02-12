import os
import scanpy as sc


def load_data(data_directory_path: str):
    file_path = os.path.join(data_directory_path, "obesity_challenge_1.h5ad")
    return sc.read_h5ad(file_path, backed="r")
