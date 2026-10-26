#import "/.calepin/calepin.typ" as calepin
#show: calepin.document

#set document(
  title: [Task №2: LDA \ Matvey Zenin 24214],
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

// Imports, settings and data reading
#calepin.chunk(
  results: "hide",
  label: "imports",
)[
  ```python
  from sklearn.datasets import make_blobs
  import pypst
  import sigfig
  import itertools as itools
  import pandas as pd
  import matplotlib as mpl
  from matplotlib import pyplot as plt 

  initial_n_samples = [1000, 1000, 1000]
  n_features = 4
  random_state = 1
  feature_names = ['f0', 'f1', 'f2', 'f3']
  class_label = 'label'
  class_to_multiply = 0

  class_to_color_map = {
      0: (0.0, 0.8, 0.0, 0.75),
      1: (0.8, 0.8, 0.0, 0.75),
      2: (0.8, 0.0, 0.0, 0.75),
  }
  class_to_marker_map = {
      0: '*',
      1: '.',
      2: 'X',
  }

  df1_X, df1_y = make_blobs(
    n_samples=initial_n_samples,
    n_features=n_features,
    random_state=random_state,
  )
  df1 = pd.DataFrame(
    df1_X, columns=feature_names
  )
  df1[class_label] = df1_y

  multiplyable_objects = df1[df1[class_label] == class_to_multiply]

  dfs = {
    1: df1,
  }
  for mul in [2, 5, 10, 20, 50, 100, 1000, 10000]:
    dfs[mul] = pd.concat([
      df1,
      multiplyable_objects.sample(frac=mul - 1, replace=True, random_state=random_state)
    ])

  def display_dataframe(df: pd.DataFrame, caption: str, include_index: bool = False):
      table = pypst.Table.from_dataframe(df, include_index = include_index)
      figure = pypst.Figure(table, caption=caption)

      print(figure.render())

  # Plot scatter diagramm with markers for each point specified
  def mscatter(x,y,ax=None, m=None, **kw):
      import matplotlib.markers as mmarkers
      if not ax: ax=plt.gca()
      sc = ax.scatter(x,y,**kw)
      if (m is not None) and (len(m)==len(x)):
          paths = []
          for marker in m:
              if isinstance(marker, mmarkers.MarkerStyle):
                  marker_obj = marker
              else:
                  marker_obj = mmarkers.MarkerStyle(marker)
              path = marker_obj.get_path().transformed(
                          marker_obj.get_transform())
              paths.append(path)
          sc.set_paths(paths)
      return sc
  ```
]

= Dataset content

#calepin.chunk(
  results: "typst",
  label: "fig-dataset-contents",
)[
  ```python
  objects_count_by_classes = (
      dfs[1]
      .groupby([class_label])
      .size()
      .reset_index(name='Objects count')
  )
  dataset_content_info = pd.DataFrame({
      "Objects amount": [len(dfs[1].index)],
      "Features amount": [len(dfs[1].columns) - 1],
      "Classes amount": [len(objects_count_by_classes.index)],
      "Objects with missing data": [
          sigfig.round(
              (
                dfs[1]
                .isna()
                .any(axis='columns')
                .sum()
              )
              / len(dfs[1].index),
              sigfigs=4,
              warn=False,
          )
      ],
  })

  display_dataframe(
    dataset_content_info,
    caption='[Main content info]',
  )
  display_dataframe(
    objects_count_by_classes,
    caption='[Distribution of objects into classes]',
  )
  ```
]

= Feature relations

#calepin.chunk(
  results: "auto",
  label: "fig-feature-relation",
  fig-caption: [Feature relations],
  fig-layout-columns: (1fr, 1fr),
)[
  ```python
  colors = list(map(
      lambda cls: class_to_color_map[cls],
      dfs[1][class_label]
  ))
  markers = list(map(
      lambda cls: class_to_marker_map[cls],
      dfs[1][class_label]
  ))

  for fst, snd in itools.combinations(feature_names, 2):
      x = dfs[1][fst]
      y = dfs[1][snd]

      fig, ax = plt.subplots()
      ax.set_axisbelow(True)
      plt.grid()
      mscatter(x, y, c=colors, m=markers, ax=ax)

      plt.xlabel(fst)
      plt.ylabel(snd)

  plt.show()
  ```
]

= Feature correlations

#calepin.chunk(
  results: "typst",
  label: "fig-feature-correlations",
)[
  ```python
  display_dataframe(
      dfs[1]
      .drop(
          class_label,
          axis='columns',
      )
      .corr(method='pearson')
      .map(
          lambda x: sigfig.round(
              x,
              sigfigs=4,
              warn=False,
          )
      ),
      caption="[Feature correlations in all dataset]",
      include_index=True,
  )
  ```
]

#calepin.chunk(
  results: "typst",
  label: "fig-feature-correlations-by-classes",
)[
  ```python
  display_dataframe(
      dfs[1]
      .groupby([class_label])
      .corr(method='pearson')
      .map(
          lambda x: sigfig.round(
              x,
              sigfigs=4,
              warn=False,
          )
      ),
      caption="[Feature correlations by groups]",
      include_index=True,
  )
  ```
]
