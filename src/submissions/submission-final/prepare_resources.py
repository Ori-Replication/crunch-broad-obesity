"""
预计算脚本：准备静态资源
本脚本在本地运行，将 Gene2vec 等静态文件复制到 resources/

Delta 预测由 train.py 使用 SVD + Ridge 在竞赛平台上训练得到。
运行方式：
    cd CrunchDAO-obesity
    conda activate broad
    python src/submissions/submission-final/prepare_resources.py
"""

import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
RESOURCES_DIR = SCRIPT_DIR / "resources"
GENE2VEC_SRC = PROJECT_ROOT / "external_repos" / "Gene2vec" / "pre_trained_emb" / "gene2vec_dim_200_iter_9_w2v.txt"

RESOURCES_DIR.mkdir(exist_ok=True, parents=True)


def main():
    print("=" * 80)
    print("Prepare Resources: Static files for SVD + Ridge submission")
    print("=" * 80, flush=True)

    if GENE2VEC_SRC.exists():
        gene2vec_dst = RESOURCES_DIR / "gene2vec_dim_200_iter_9_w2v.txt"
        shutil.copy2(GENE2VEC_SRC, gene2vec_dst)
        print(f"  Copied Gene2vec to: {gene2vec_dst}", flush=True)
    else:
        print(f"  Warning: Gene2vec not found at {GENE2VEC_SRC}", flush=True)
        print("  train.py will use fallback path if available.", flush=True)

    print("\n" + "=" * 80)
    print("Resources prepared. Run train() on platform to generate SVD+Ridge model.")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
