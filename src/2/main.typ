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

#let py = calepin.inline.with("python")

#align(center, title())

// Imports, settings and data reading
#calepin.chunk(
  results: "hide",
  label: "imports",
)[
  ```python
  from sklearn.svm import LinearSVC
  from matplotlib.colors import ListedColormap
  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
  from sklearn.inspection import DecisionBoundaryDisplay
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

  class_to_multiply = 2
  secondary_class = 0

  lda_features = ('f0', 'f1')

  class_to_color_map = {
      2: (0.0, 0.8, 0.0, 0.75),
      1: (0.8, 0.8, 0.0, 0.75),
      0: (0.8, 0.0, 0.0, 0.75),
  }
  class_to_marker_map = {
      2: '*',
      1: '.',
      0: 'X',
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

= LDA/SVM

For classification of objects with LDA/SVM let's filter out all classes except for:
    #calepin.chunk(
      results: "typst",
      label: "lda-svm-classes-list",
      echo: false,
    )[
      ```python
      print(pypst.Itemize([str(class_to_multiply), str(secondary_class)]).render())
      ```
    ]
And all features except for:
    #calepin.chunk(
      results: "typst",
      label: "lda-svm-features-list",
      echo: false,
    )[
      ```python
      print(pypst.Itemize(list(lda_features)).render())
      ```
    ]

#calepin.chunk(
  results: "hide",
  label: "lda-svm-filtering",
)[
  ```python
  for k, v in dfs.items():
    v.drop(
      filter(
        lambda c: c != class_label and c not in lda_features,
        v.columns,
      ),
      axis='columns',
      inplace=True,
    )

    dfs[k] = v[v[class_label].map(lambda c: c == class_to_multiply or c == secondary_class)]
  ```
]

#calepin.chunk(
  results: "auto",
  label: "fig-lda",
  fig-caption: [LDA],
  fig-layout-columns: (1fr, 1fr),
  fig-subcaptions: (
    [df1],
    [df2],
    [df5],
    [df10],
    [df20],
    [df50],
    [df100],
    [df1000],
    [df10k],
  ),
)[
  ```python
  X_df1 = dfs[1][list(lda_features)]
  y_df1 = dfs[1][class_label]
  colors = y_df1.map(lambda q: class_to_color_map[q])
  markers = y_df1.map(lambda q: class_to_marker_map[q])

  lda_clfs = {}

  for k,df in dfs.items():
    X = df[list(lda_features)]
    y = df[class_label]

    lda_clfs[k] = LinearDiscriminantAnalysis().fit(X, y)

    cmap = ListedColormap([class_to_color_map[c] for c in lda_clfs[k].classes_])
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    plt.grid()

    DecisionBoundaryDisplay.from_estimator(
        lda_clfs[k], X, response_method="predict",
        eps=1.0,
        ax=ax,
        xlabel=lda_features[0], ylabel=lda_features[1],
        alpha=0.2,
        cmap=cmap,
        grid_resolution=256,
    )
    mscatter(X_df1[lda_features[0]], X_df1[lda_features[1]], ax=ax,  c=colors, m=markers)
    plt.show()
  ```
]

#calepin.chunk(
  results: "auto",
  label: "fig-svm",
  fig-caption: [SVM],
  fig-layout-columns: (1fr, 1fr),
  fig-subcaptions: (
    [df1],
    [df2],
    [df5],
    [df10],
    [df20],
    [df50],
    [df100],
    [df1000],
    [df10k],
  ),
)[
  ```python
  X_df1 = dfs[1][list(lda_features)]
  y_df1 = dfs[1][class_label]
  colors = y_df1.map(lambda q: class_to_color_map[q])
  markers = y_df1.map(lambda q: class_to_marker_map[q])

  lda_clfs = {}

  for k,df in dfs.items():
    X = df[list(lda_features)]
    y = df[class_label]

    clf = LinearSVC().fit(X, y)

    cmap = ListedColormap([class_to_color_map[c] for c in clf.classes_])
    fig, ax = plt.subplots()
    ax.set_axisbelow(True)
    plt.grid()

    DecisionBoundaryDisplay.from_estimator(
        clf, X, response_method="predict",
        eps=1.0,
        ax=ax,
        xlabel=lda_features[0], ylabel=lda_features[1],
        alpha=0.2,
        cmap=cmap,
        grid_resolution=256,
    )
    mscatter(X_df1[lda_features[0]], X_df1[lda_features[1]], ax=ax,  c=colors, m=markers)
    plt.show()
  ```
]
