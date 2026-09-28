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
import numpy as np
import pandas as pd
import pypst
import sigfig
import itertools as itools
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, auc, roc_curve, RocCurveDisplay
from sklearn.calibration import CalibratedClassifierCV
import matplotlib as mpl
from matplotlib import pyplot as plt 

features = ['residual sugar', 'pH', 'alcohol', 'fixed acidity']
mid_quality_cutoff = 7
poor_quality_cutoff = 5
lda_classes = ['great', 'poor']
linear_regression_features = ('pH', 'alcohol')
classifiers_comparison_classes = ['great', 'poor']
classifiers_comparison_features = ('pH', 'alcohol')
classifiers_comparison_target_class = 'great'

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

def display_classifier(clf, X, y, colors, markers):
    disp = DecisionBoundaryDisplay.from_estimator(
        clf, X, response_method="predict",
        xlabel=classifiers_comparison_features[0], ylabel=classifiers_comparison_features[1],
        alpha=0.5,
        colors=colors,
        grid_resolution=256,
    )
    mscatter(X[classifiers_comparison_features[0]], X[classifiers_comparison_features[1]], ax=disp.ax_,  c=colors, m=markers, edgecolor="k")
    plt.show()

def display_classifier_metrics(clf, target_class, X, y, clf_name):
    class_mapper = lambda cl: cl if cl == target_class else 'not ' + target_class

    y = list(map(class_mapper, y))
    y_pred = list(map(class_mapper, clf.predict(X)))

    tp, fn, fp, tn = confusion_matrix(
        y, y_pred, 
        labels=[target_class, 'not ' + target_class],
    ).ravel().tolist()

    display_dataframe(
        pd.DataFrame(
            {
                'Predicted ' + target_class: [tp, fp],
                'Predicted not ' + target_class: [fn, tn],
            },
            index=[target_class, 'not ' + target_class],
        ),
        caption=f"[{clf_name} confusion matrix]",
        include_index=True,
    )

    print(pypst.Itemize([
      f"Sensitivity: {sigfig.round(tp / (tp + fn), sigfigs=4, warn=False)}",
      f"Specificity: {sigfig.round(tn / (fp + tn), sigfigs=4, warn=False)}",
      f"Precision: {sigfig.round(tp / (tp + fp), sigfigs=4, warn=False)}",
      f"Recall: {sigfig.round(tp / (tp + fn), sigfigs=4, warn=False)}",
    ]).render())

    fpr, tpr, thresholds = roc_curve(
      y,
      clf.predict_proba(X)[:, np.where(clf.classes_ == target_class)[0]], 
      pos_label=target_class,
    )
    roc_auc = auc(fpr, tpr)
    display = RocCurveDisplay(
      fpr=fpr, tpr=tpr, roc_auc=roc_auc,
      name=clf_name, pos_label=target_class,
    )
    display.plot()

    plt.show()

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

= LDA

For classification of objects with LDA let's filter out all classes except for:
    #calepin.chunk(
      echo: false,
      results: "typst",
    )[
      ```python
      print(pypst.Itemize(lda_classes).render())
      ```
    ]

#calepin.chunk(
  echo: false,
  results: "auto",
  label: "fig-lda",
  fig-caption: [LDA for every pair of features],
  fig-layout-columns: (1fr),
)[
  ```python
  lda_data = data[data['quality'].map(lambda q: q in lda_classes)]

  y = lda_data['quality']
  colors = y.map(lambda q: class_to_color_map[q])
  markers = y.map(lambda q: class_to_marker_map[q])

  for fst, snd in itools.combinations(features, 2):
      X = lda_data[[fst, snd]]
      clf = LinearDiscriminantAnalysis().fit(X, y)

      disp = DecisionBoundaryDisplay.from_estimator(
          clf, X, response_method="predict",
          xlabel=fst, ylabel=snd,
          alpha=0.5,
          colors=colors,
          grid_resolution=256,
      )
      mscatter(X[fst], X[snd], ax=disp.ax_,  c=colors, m=markers, edgecolor="k")

      plt.show()
  ```
]

= Linear regression + LDA

#calepin.chunk(
  echo: false,
  results: "auto",
  label: "fig-lr-lda",
  fig-caption: [Linear regression + LDA],
)[
  ```python
  fig, ax = plt.subplots()

  X = lda_data[[linear_regression_features[0], linear_regression_features[1]]]
  y = lda_data['quality']
  colors = y.map(lambda q: class_to_color_map[q])
  markers = y.map(lambda q: class_to_marker_map[q])

  clf = LinearDiscriminantAnalysis().fit(X, y)

  disp = DecisionBoundaryDisplay.from_estimator(
      clf, X, response_method="predict",
      xlabel=linear_regression_features[0], ylabel=linear_regression_features[1],
      alpha=0.5,
      ax=ax,
      colors=colors,
      grid_resolution=256,
  )
  mscatter(X[linear_regression_features[0]], X[linear_regression_features[1]], ax=ax,  c=colors, m=markers, edgecolor="k")

  lr = LinearRegression().fit(X[[linear_regression_features[0]]], X[linear_regression_features[1]])
  left, right = ax.get_xlim()
  x = pd.DataFrame({
      linear_regression_features[0]: np.linspace(left, right, num=256)
  })
  y = lr.predict(x)
  ax.plot(x, y, color=(0.0, 0.0, 0.0))

  plt.show()
  ```
]

= Classification methods comparison

For comparison of classification methods let's filter out all classes except for:
    #calepin.chunk(
      echo: false,
      results: "typst",
    )[
      ```python
      print(pypst.Itemize(classifiers_comparison_classes).render())
      ```
    ]

Also, for classification let's use only following features:
    #calepin.chunk(
      echo: false,
      results: "typst",
    )[
      ```python
      print(pypst.Itemize(list(classifiers_comparison_features)).render())
      ```
    ]

The target class for classification will be #py[`classifiers_comparison_target_class`] and methods which will be compared are:
- LDA
- SVM
- Logistic regression
- Naive Bayes classifier

```python
#| results: hide
#| echo: false

comp_data = data[data['quality'].map(lambda q: q in classifiers_comparison_classes)]

X = comp_data[[classifiers_comparison_features[0], classifiers_comparison_features[1]]]
y = comp_data['quality']
colors = y.map(lambda q: class_to_color_map[q])
markers = y.map(lambda q: class_to_marker_map[q])
```

== LDA

#calepin.chunk(
  echo: false,
  results: "auto",
  label: "fig-lr-lda2",
  fig-caption: [LDA],
  fig-layout-columns: (1fr),
)[
  ```python
  clf = LinearDiscriminantAnalysis().fit(X, y)
  display_classifier(clf, X, y, colors, markers)
  ```
]

#calepin.chunk(
  echo: false,
  results: "typst",
)[
  ```python
  display_classifier_metrics(clf, classifiers_comparison_target_class, X, y, "LDA")
  ```
]

== SVM

#calepin.chunk(
  echo: false,
  results: "auto",
  label: "fig-lr-svm",
  fig-caption: [SVM],
  fig-layout-columns: (1fr),
)[
  ```python
  clf = make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(gamma='auto'), ensemble=False)).fit(X, y)
  display_classifier(clf, X, y, colors, markers)
  ```
]

#calepin.chunk(
  echo: false,
  results: "typst",
)[
  ```python
  display_classifier_metrics(clf, classifiers_comparison_target_class, X, y, "SVM")
  ```
]

== Logistic regression

#calepin.chunk(
  echo: false,
  results: "auto",
  label: "fig-lr-log",
  fig-caption: [Logistic regression],
  fig-layout-columns: (1fr),
)[
  ```python
  clf = LogisticRegression().fit(X, y)
  display_classifier(clf, X, y, colors, markers)
  ```
]

#calepin.chunk(
  echo: false,
  results: "typst",
)[
  ```python
  display_classifier_metrics(clf, classifiers_comparison_target_class, X, y, "Logistic regression")
  ```
]

== Naive Bayes classifier

#calepin.chunk(
  echo: false,
  results: "auto",
  label: "fig-lr-bayes",
  fig-caption: [Naive Bayes classifier],
  fig-layout-columns: (1fr),
)[
  ```python
  clf = GaussianNB().fit(X, y)
  display_classifier(clf, X, y, colors, markers)
  ```
]

#calepin.chunk(
  echo: false,
  results: "typst",
)[
  ```python
  display_classifier_metrics(clf, classifiers_comparison_target_class, X, y, "Naive Bayes classifier")
  ```
]
