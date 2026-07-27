"""BREST: bulk-RNA -> spatial transcriptomics with a unified graph-transformer.

One architecture (H0-mini -> Delaunay graph -> GAT + global Transformer ->
gene_queries head) shared across bulk / Visium / Xenium, so a bulk backbone
transfers 1:1 into the spatial models by gene symbol.
"""
__version__ = "0.1.0"
