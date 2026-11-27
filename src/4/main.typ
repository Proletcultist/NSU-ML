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
  from sklearn.preprocessing import StandardScaler
  from sklearn.decomposition import PCA
  from itertools import combinations

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

= Plotting 4 components that explain the greatest amount of variance

For `df_4`:

#calepin.chunk(
  results: "auto",
  label: "fig-explanatory-comps",
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

  X = dfs[4]
  X_scaled = StandardScaler().fit_transform(X)
  pca = PCA()
  scores = pca.fit_transform(X_scaled)
  loadings = pca.components_

  plot_pc_pairs(scores, loadings, [0, 1, 2, 3], show_vectors=False)
  ```
]

#calepin.chunk(
  results: "auto",
  label: "fig-explanatory-comps-biplot",
  fig-caption: [df_4 pairs of components 0, 1, 3, 4 on biplot],
  fig-layout-columns: (1fr, 1fr),
)[
  ```python
  plot_pc_pairs(scores, loadings, [0, 1, 2, 3], show_vectors=True)
  ```
]
