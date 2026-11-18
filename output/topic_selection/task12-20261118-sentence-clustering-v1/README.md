# Task12 sentence-level clustering

This directory contains KMeans candidates on the delivered sentence-level BGE-M3 vectors. The fit population is the 7,269-unit `fit_candidate=true` subset. The remaining 1,710 units are assigned to nearest centroids for application-only inspection. Cluster IDs are not final topic IDs. Five 80% stratified resamples per candidate are reported in `resampling_results.json`. Semantic review against `topic-list-draft.json` is still required before freezing `topics-v1`.
