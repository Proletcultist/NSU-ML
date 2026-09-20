#import "/.calepin/calepin.typ" as calepin
#show: calepin.document

#set document(
  title: [Task №1: Wine quality dataset analysis \ Matvey Zenin 24214],
)
#set page(
  paper: "a4",
  numbering: "1",
)
#set text(
  font: "New Computer Modern",
  size: 12pt,
)
#show figure: set block(breakable: true)
#let py = calepin.inline.with("python")

// Imports, settings and data reading
```python
#| results: hide
#| echo: false
import pandas as pd
import pypst
import sigfig
import itertools as itools
from matplotlib import pyplot as plt 

poor_quality_cutoff = 5
mid_quality_cutoff = 7
features = ['residual sugar', 'pH', 'alcohol','fixed acidity']

data: pd.DataFrame = pd.read_csv('data.csv')

def display_dataframe(df: pd.DataFrame, caption: str, include_index: bool = False):
    table = pypst.Table.from_dataframe(df, include_index = include_index)
    figure = pypst.Figure(table, caption=caption)

    print(figure.render())

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

#align(center, title())

= Dataset content

#calepin.chunk(
  echo: false,
  results: "typst",
)[
  ```python
  def get_dataset_content_info() -> (pd.DataFrame, pd.DataFrame):
      objects_count_by_classes = (
          data
          .groupby(['quality'])
          .size()
          .reset_index(name='Objects count')
      )
      dataset_content_info = pd.DataFrame({
          "Objects amount": [len(data.index)],
          "Features amount": [len(data.columns) - 1],
          "Classes amount": [len(objects_count_by_classes.index)],
          "Objects with missing data": [
              sigfig.round(
                  (
                    data
                    .isna()
                    .any(axis='columns')
                    .sum()
                  )
                  / len(data.index),
                  sigfigs=4,
                  warn=False,
              )
          ],
      })
      return (dataset_content_info, objects_count_by_classes)

  dataset_content_info, objects_count_by_classes = get_dataset_content_info()
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

= Dataset filtering

Let's transform dataset by:
  + Transforming quality feature:
    - If $"quality" < #py[`poor_quality_cutoff`]$ the quality is "poor"
    - If $#py[`poor_quality_cutoff`] <= "quality" < #py[`mid_quality_cutoff`]$ the quality is "mid"
    - If $#py[`mid_quality_cutoff`] <= "quality"$ the quality is "great"
  + Excluding all features except for:
    #calepin.chunk(
      echo: false,
      results: "typst",
    )[
      ```python
      print(pypst.Itemize(features).render())
      ```
    ]
  + Removing all objects with missing data

#calepin.chunk(
  echo: false,
  results: "typst",
)[
  ```python
  data['quality'] = (
      data['quality']
      .map(lambda x: 
        "poor" if x < poor_quality_cutoff else
        "mid" if x >= poor_quality_cutoff and x < mid_quality_cutoff else
        "great" if x >= mid_quality_cutoff else
        None
      )
  )
  data.drop(
      filter(
          lambda c: c not in ['quality'] + features, 
          data.columns,
      ),
      axis='columns',
      inplace=True,
  )
  data.dropna(inplace=True)

  dataset_content_info, objects_count_by_classes = get_dataset_content_info()
  display_dataframe(
      dataset_content_info, 
      caption='[Main content info after transformation]',
  )
  display_dataframe(
      objects_count_by_classes,
      caption='[Distribution of objects into classes after transformation]',
  )
  ```
]

= Feature relations

#calepin.chunk(
  echo: false,
  results: "auto",
  label: "fig-feature-relation",
  fig-caption: [Feature relations],
  fig-layout-columns: (1fr),
)[
  ```python
  class_to_color_map = {
      'great': (0.0, 0.8, 0.0),
      'mid': (0.8, 0.8, 0.0),
      'poor': (0.8, 0.0, 0.0),
  }
  class_to_marker_map = {
      'great': '*',
      'mid': '.',
      'poor': 'X',
  }

  colors = list(map(
      lambda cls: class_to_color_map[cls],
      data['quality']
  ))
  markers = list(map(
      lambda cls: class_to_marker_map[cls],
      data['quality']
  ))

  for fst, snd in itools.combinations(features, 2):
      x = data[fst]
      y = data[snd]

      fig, ax = plt.subplots()
      mscatter(x, y, c=colors, m=markers, ax = ax)

      plt.xlabel(fst)
      plt.ylabel(snd)

  plt.show()
  ```
]

#calepin.chunk(
  echo: false,
  results: "auto",
  label: "fig-features-hist",
  fig-caption: [Objects count by feature values],
  fig-layout-columns: (1fr),
)[
  ```python
  for feat in features:
      fig, ax = plt.subplots()

      ax.hist(data[feat], bins=20)

      plt.xlabel(feat)
      plt.ylabel('Objects count')

  plt.show()
  ```
]

= Feature correlations

#calepin.chunk(
  echo: false,
  results: "typst",
)[
  ```python
  display_dataframe(
      data
      .drop(
          'quality',
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
  echo: false,
  results: "typst",
)[
  ```python
  display_dataframe(
      data
      .groupby(['quality'])
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
