#import "/.calepin/calepin.typ" as calepin
#show: calepin.document

#set document(
  title: [Task №4: PCA, tSNE, UMAP \ Matvey Zenin 24214],
)
#set page(
  paper: "a4",
  numbering: none,
)
#set text(
  font: "New Computer Modern",
  size: 12pt,
)
#calepin.setup(echo: true)

#show figure: set block(breakable: true)
#let py = calepin.inline.with("python")

#align(center, title())

#calepin.chunk(
  results: "hide",
  label: "imports",
)[
  ```python
  import numpy as np
  import matplotlib.pyplot as plt
  import matplotlib.patches as mpatches
  from sklearn.manifold import TSNE
  from umap import UMAP
  from pacmap import PaCMAP
  from sklearn.preprocessing import StandardScaler
  from sklearn.decomposition import PCA
  from itertools import combinations

  low_components_to_show = [0, 1, 2, 3]
  high_components_to_show = [129, 130, 131, 132]
  random_states = [1337, 222]

  dfs = {}
  for a in [4, 16, 256]:
    dfs[a] = np.zeros((a, a), dtype=int)
    length = a // 2
    for x in range(a):
        dfs[a][x, x : min(x + length, a)] = 1
  ```
]

= 4x4, 16x16, 256x256 datasets visualization

#calepin.chunk(
  results: "auto",
  label: "fig-dfs-vis",
  fig-caption: [df_a visualizations],
  fig-layout-columns: (1fr, 1fr),
  fig-subcaptions: (
    [df_4],
    [df_16],
    [df_256],
  ),
)[
  ```python
  for _, df in dfs.items():
    fig, ax = plt.subplots()

    ax.imshow(df, cmap='Greys', interpolation='nearest')
    ax.set_xlabel('Y')
    ax.set_ylabel('X')

    plt.show()
  ```
]

= Scree plot

#calepin.chunk(
  results: "auto",
  label: "fig-scree",
  fig-caption: [Scree plots],
  fig-layout-columns: (1fr, 1fr),
  fig-subcaptions: (
    [df_4],
    [df_16],
    [df_256],
  ),
)[
  ```python
  dfs_scaled = {}
  pcas = {}
  for a in dfs:
    dfs_scaled[a] = StandardScaler().fit_transform(dfs[a])
    pcas[a] = PCA().fit(dfs_scaled[a])

  for a in dfs:
    fig, ax = plt.subplots()
    ax.plot(range(1, len(pcas[a].explained_variance_ratio_) + 1), pcas[a].explained_variance_ratio_, marker='o', markersize=3)
    plt.xlabel("Component num")
    plt.ylabel("Share of explained variance")
    ax.grid(True)
    plt.show()
  ```
]

= Plotting 4 components that explain the greatest amount of variance

== `df_4`

#calepin.chunk(
  results: "auto",
  label: "fig-df-4-explanatory-comps",
  fig-caption: [df_4 pairs of components 0, 1, 3, 4],
  fig-layout-columns: (1fr, 1fr),
)[
  ```python
  def plot_pc_pairs(scores, loadings, comp_indices, show_vectors=False, title=""):
    pairs = list(combinations(comp_indices, 2))
    n_pairs = len(pairs)
    cols = 3
    rows = (n_pairs + cols - 1) // cols
    
    for idx, (i, j) in enumerate(pairs):
      fig, ax = plt.subplots()

      ax.scatter(scores[:, i], scores[:, j])
      
      if show_vectors:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        half_span_x = 0.4 * (xmax - xmin)
        half_span_y = 0.4 * (ymax - ymin)
        
        max_load_x = np.max(np.abs(loadings[i, :])) or 1.0
        max_load_y = np.max(np.abs(loadings[j, :])) or 1.0
        
        for k in range(loadings.shape[1]):
            dx = (loadings[i, k] / max_load_x) * half_span_x
            dy = (loadings[j, k] / max_load_y) * half_span_y

            arrow = mpatches.FancyArrowPatch((cx, cy), (cx + dx, cy + dy),
                                 mutation_scale=12)
            ax.add_patch(arrow)
              
      ax.set_xlabel(f"PC {i+1}")
      ax.set_ylabel(f"PC {j+1}")
      ax.grid(True, alpha=0.3)

      plt.show()

  scores = pcas[4].transform(dfs_scaled[4])
  loadings = pcas[4].components_
  plot_pc_pairs(scores, loadings, low_components_to_show, show_vectors=False)
  ```
]

#calepin.chunk(
  results: "auto",
  label: "fig-df-4-explanatory-comps-biplot",
  fig-caption: [df_4 pairs of components 0, 1, 3, 4 on biplot],
  fig-layout-columns: (1fr, 1fr),
)[
  ```python
  plot_pc_pairs(scores, loadings, low_components_to_show, show_vectors=True)
  ```
]

== `df_16`

#calepin.chunk(
  results: "auto",
  label: "fig-df-16-explanatory-comps",
  fig-caption: [df_16 pairs of components 0, 1, 3, 4],
  fig-layout-columns: (1fr, 1fr),
)[
  ```python
  scores = pcas[16].transform(dfs_scaled[16])
  loadings = pcas[16].components_
  plot_pc_pairs(scores, loadings, low_components_to_show, show_vectors=False)
  ```
]

#calepin.chunk(
  results: "auto",
  label: "fig-df-16-explanatory-comps-biplot",
  fig-caption: [df_16 pairs of components 0, 1, 3, 4 on biplot],
  fig-layout-columns: (1fr, 1fr),
)[
  ```python
  plot_pc_pairs(scores, loadings, low_components_to_show, show_vectors=True)
  ```
]

== `df_256`

#calepin.chunk(
  results: "auto",
  label: "fig-df-256-explanatory-comps",
  fig-caption: [df_256 pairs of components 0, 1, 3, 4],
  fig-layout-columns: (1fr, 1fr),
)[
  ```python
  scores = pcas[256].transform(dfs_scaled[256])
  loadings = pcas[256].components_
  plot_pc_pairs(scores, loadings, low_components_to_show, show_vectors=False)
  ```
]

#calepin.chunk(
  results: "auto",
  label: "fig-df-256-explanatory-comps-biplot",
  fig-caption: [df_256 pairs of components 0, 1, 3, 4 on biplot],
  fig-layout-columns: (1fr, 1fr),
)[
  ```python
  plot_pc_pairs(scores, loadings, low_components_to_show, show_vectors=True)
  ```
]

= Plotting 4 components for df_256 that explain the least amount of variance

#calepin.chunk(
  results: "auto",
  label: "fig-df-256-not-explanatory-comps",
  fig-caption: [df_256 pairs of components 129, 130, 131, 132],
  fig-layout-columns: (1fr, 1fr),
)[
  ```python
  plot_pc_pairs(scores, loadings, high_components_to_show, show_vectors=False)
  ```
]

#calepin.chunk(
  results: "auto",
  label: "fig-df-256-not-explanatory-comps-biplot",
  fig-caption: [df_256 pairs of components 129, 130, 131, 132 on biplot],
  fig-layout-columns: (1fr, 1fr),
)[
  ```python
  plot_pc_pairs(scores, loadings, high_components_to_show, show_vectors=True)
  ```
]

= Visualization of df_a datasets with UMAP

#calepin.chunk(
  results: "auto",
  label: "fig-umap-df-a-1",
  fig-caption: [df_a datasets UMAP visualization],
  fig-layout-columns: (1fr, 1fr),
  fig-subcaptions: (
    [df_4],
    [df_16],
    [df_256],
  ),
)[
  ```python
  for a in dfs_scaled:
    embedding = UMAP(
      n_neighbors=min(15, a - 1),
      compatibility_layout=False,
      random_state=random_states[0],
    ).fit_transform(dfs_scaled[a])

    fig, ax = plt.subplots()
    ax.scatter(embedding[:, 0], embedding[:, 1])
    plt.show()
  ```
]

#calepin.chunk(
  results: "auto",
  label: "fig-umap-df-a-2",
  fig-caption: [df_a datasets UMAP visualization 2],
  fig-layout-columns: (1fr, 1fr),
  fig-subcaptions: (
    [df_4],
    [df_16],
    [df_256],
  ),
)[
  ```python
  for a in dfs_scaled:
    embedding = UMAP(
      n_neighbors=min(15, a - 1),
      compatibility_layout=False,
      random_state=random_states[1],
    ).fit_transform(dfs_scaled[a])

    fig, ax = plt.subplots()
    ax.scatter(embedding[:, 0], embedding[:, 1])
    plt.show()
  ```
]

= Visualization of df_a datasets with PacMAP

#calepin.chunk(
  results: "auto",
  label: "fig-pacmap-df-a-1",
  warning: false,
  fig-caption: [df_a datasets PacMAP visualization],
  fig-layout-columns: (1fr, 1fr),
  fig-subcaptions: (
    [df_4],
    [df_16],
    [df_256],
  ),
)[
  ```python
  for a in dfs_scaled:
    embedding = PaCMAP(
      n_neighbors=min(10, a - 1),
      random_state=random_states[0],
    ).fit_transform(dfs_scaled[a])

    fig, ax = plt.subplots()
    ax.scatter(embedding[:, 0], embedding[:, 1])
    plt.show()
  ```
]

#calepin.chunk(
  results: "auto",
  label: "fig-pacmap-df-a-2",
  warning: false,
  fig-caption: [df_a datasets PacMAP visualization 2],
  fig-layout-columns: (1fr, 1fr),
  fig-subcaptions: (
    [df_4],
    [df_16],
    [df_256],
  ),
)[
  ```python
  for a in dfs_scaled:
    embedding = PaCMAP(
      n_neighbors=min(10, a - 1),
      random_state=random_states[1],
    ).fit_transform(dfs_scaled[a])

    fig, ax = plt.subplots()
    ax.scatter(embedding[:, 0], embedding[:, 1])
    plt.show()
  ```
]

= Visualization of df_a datasets with tSNE

#calepin.chunk(
  results: "auto",
  label: "fig-tsne-df-a-1",
  fig-caption: [df_a datasets tSNE visualization],
  fig-layout-columns: (1fr, 1fr),
  fig-subcaptions: (
    [df_4],
    [df_16],
    [df_256],
  ),
)[
  ```python
  for a in dfs_scaled:
    embedding = TSNE(
      perplexity=min(30, a - 1),
      random_state=random_states[0],
      learning_rate='auto'
    ).fit_transform(dfs_scaled[a])

    fig, ax = plt.subplots()
    ax.scatter(embedding[:, 0], embedding[:, 1])
    plt.show()
  ```
]

#calepin.chunk(
  results: "auto",
  label: "fig-tsne-df-a-2",
  fig-caption: [df_a datasets tSNE visualization 2],
  fig-layout-columns: (1fr, 1fr),
  fig-subcaptions: (
    [df_4],
    [df_16],
    [df_256],
  ),
)[
  ```python
  for a in dfs_scaled:
    embedding = TSNE(
      perplexity=min(30, a - 1),
      random_state=random_states[1],
      learning_rate='auto'
    ).fit_transform(dfs_scaled[a])

    fig, ax = plt.subplots()
    ax.scatter(embedding[:, 0], embedding[:, 1])
    plt.show()
  ```
]
