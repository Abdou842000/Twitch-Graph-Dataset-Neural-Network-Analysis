# Evaluating GNNs on Twitch Gamers Dataset

## Project Overview

This project evaluates the effectiveness of Graph Neural Networks (GNNs) on the **Twitch Gamers dataset**, focusing on tasks like explicit content identification, affiliate status prediction, language classification, and views prediction. We benchmark modern GNN models against traditional methods, highlighting their performance on graph-structured data.

## Objectives

- Apply GNN models to solve four key tasks:
  1. **Explicit Content Identification**
  2. **Affiliate Status Prediction**
  3. **Language Classification**
  4. **Views Prediction**
- Compare GNN models' performance with classical models (e.g., XGBoost).

## Dataset

- **Twitch Gamers Dataset (2018)**:
  - Contains ~168K nodes and 6.79M edges.
  - Node attributes: ID, activity, language, affiliate status, explicit content, views, and account lifetime.
  - An undirected graph representing mutual connections between streamers.

## Models Used

- **Graph Convolutional Network (GCN)**
- **Graph Isomorphism Network (GIN)**
- **Graph Attention Network (GAT)**
- **GraphSAGE (Sample and Aggregate)**

## Results Summary

- **Explicit Content Identification**: GNN models outperformed XGB, with GAT and GraphSAGE showing the highest accuracy (64%).
- **Affiliate Status Prediction**: GraphSAGE achieved the best recall (84%), outperforming other models.
- **Language Classification**: GAT outperformed other models with 90% accuracy.
- **Log-Views Regression**: GraphSAGE showed the lowest MSE and highest R-squared score, indicating superior performance.

## Conclusion

GNN models demonstrated a significant improvement over traditional methods for handling complex graph data. However, there is still room for advancement with newer architectures. This project highlights the Twitch Gamers dataset as an ideal benchmark for GNN evaluation.

---

Project by: Gaspard Berthelier, Guillaume Levy, Abderrahim Namouh, Clement Wang  
