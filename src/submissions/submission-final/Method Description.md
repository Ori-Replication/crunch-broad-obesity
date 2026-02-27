# Method Description

This method employs **SVD + Ridge regression** to predict gene perturbation effects in adipocyte precursor cells, using only the competition training data without external datasets. For the Pearson Delta metric, this SVD-based approach achieves the best performance among the methods explored.

## Core Approach: SVD Gene Embedding + Ridge Delta Regression

1. **Gene embedding via TruncatedSVD**: Apply TruncatedSVD (n_components=50) on NC (control) cell expression matrix. The transpose of SVD components yields gene embeddings of shape (n_genes, 50), capturing low-dimensional gene representations from the control transcriptome.

2. **Ridge regression for Delta prediction**: For each training perturbation, compute the mean expression delta (perturbed_mean - control_mean). Train a Ridge regression model (alpha=10.0) that maps gene embedding to full delta vector. The model learns the mapping from gene identity (in SVD space) to perturbation effect.

3. **Inference**: For each target perturbation, retrieve its gene embedding (or use the mean embedding for genes not in the training set). Predict delta via the trained Ridge model. Add the predicted delta to sampled NC cells to generate perturbed cell profiles.

4. **Cell sampling**: Sample 100 NC cells with mean-matching (n_trials=1000) so that the sampled mean closely approximates the overall control mean, preserving distributional properties.

5. **Cell proportion prediction**: Use Gene2vec KNN predictor (ProportionKNNPredictor, K=15) to predict pre_adipo, adipo, lipo, other proportions. Genes without Gene2vec embedding are represented by zero vectors. Fallback to average proportions when the predictor is unavailable.

## Rationale

The SVD-based method projects genes into a low-dimensional space derived from the control transcriptome structure. Ridge regression then learns a linear mapping from this space to perturbation effects. This approach is robust with limited training samples (~122 perturbations) and achieves strong Pearson Delta scores. Cell sampling with mean-matching and Gene2vec KNN proportion prediction are retained from prior experiments for MMD and L1 metrics.

## Data and Resources Used

- **Competition data only**: `obesity_challenge_1.h5ad` for SVD fitting, Ridge training, and NC cell sampling. No external datasets (e.g., xatlas) are used.
- **Gene2vec**: Pre-trained gene embeddings (`gene2vec_dim_200_iter_9_w2v.txt`) for ProportionKNNPredictor. Prepared by `prepare_resources.py`.
- **Program proportions**: Gene2vec KNN predictor trained on `program_proportion.csv`; fallback to average proportions.
