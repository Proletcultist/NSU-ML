#import "/.calepin/calepin.typ" as calepin
#show: calepin.document

#set document(
  title: [Task №3: LogReg \ Matvey Zenin 24214],
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
  from matplotlib.patches import Rectangle
  import scikits.bootstrap as boot
  from sklearn.metrics import confusion_matrix, auc, roc_curve, RocCurveDisplay
  import numpy as np
  from sklearn.linear_model import LinearRegression, LogisticRegression
  from sklearn.datasets import make_blobs
  from sklearn.model_selection import train_test_split
  import pandas as pd
  import pypst
  import itertools as itools
  import sigfig
  import matplotlib as mpl
  from matplotlib import pyplot as plt 

  initial_n_samples = [100, 100, 100]
  n_features = 4
  class_centers = [
    [1, 1, 1, 1],
    [-1, -1, -1, -1],
    [-1, 1, -1, 1],
  ]
  feature_names = ['f0', 'f1', 'f2', 'f3']
  class_label = 'label'
  random_state = 1

  class_to_enlarge = 0
  secondary_class = 1
  regression_feature = 'f0'
  test_size = 0.25
  boot_n = 1000
  confidence_level = 0.95

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

  df_X, df_y = make_blobs(
    n_samples=initial_n_samples,
    n_features=n_features,
    centers=class_centers,
    random_state=random_state,
  )
  df = pd.DataFrame(
    df_X, columns=feature_names
  )
  df[class_label] = df_y

  dfs = {}
  dfs[0] = {}
  dfs[0][0] = df
  for amount in [1, 10, 100]:
    dfs[amount] = {}
    for cord in [5, 10, 20]:
      dfs[amount][cord] = pd.concat(
        [df] +
        [pd.DataFrame({
          feature_names[0]: [cord],
          feature_names[1]: [-cord],
          feature_names[2]: [cord],
          feature_names[3]: [-cord],
          class_label: class_to_enlarge,
        })] * amount, 
        ignore_index=True)

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
      df
      .groupby([class_label])
      .size()
      .reset_index(name='Objects count')
  )
  dataset_content_info = pd.DataFrame({
      "Objects amount": [len(df.index)],
      "Features amount": [len(df.columns) - 1],
      "Classes amount": [len(objects_count_by_classes.index)],
      "Objects with missing data": [
          sigfig.round(
              (
                df
                .isna()
                .any(axis='columns')
                .sum()
              )
              / len(df.index),
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
      df[class_label]
  ))
  markers = list(map(
      lambda cls: class_to_marker_map[cls],
      df[class_label]
  ))

  for fst, snd in itools.combinations(feature_names, 2):
      x = df[fst]
      y = df[snd]

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
      df
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
      df
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

= Classification with LDA and LogReg

#calepin.chunk(
  results: "hide",
  label: "lda-logreg-functions",
)[
  ```python
  def plot_classifier_roc_auc(clf, target_class, X_test, y_test, clf_name, ax):
      fpr, tpr, thresholds = roc_curve(
        y_test,
        clf.predict_proba(X_test)[:, np.where(clf.classes_ == target_class)[0]], 
        pos_label=target_class,
      )
      roc_auc = auc(fpr, tpr)
      display = RocCurveDisplay(
        fpr=fpr, tpr=tpr, roc_auc=roc_auc,
        name=clf_name, pos_label=target_class,
      )
      display.plot(ax=ax)

  def calc_specificity_sensitivity_point(clf, target_class, X_test, y_test):
    y_true = np.asarray(y_test).ravel() == target_class
    y_pred = np.asarray(clf.predict(X_test)).ravel() == target_class

    tp, fn, fp, tn = confusion_matrix(
        y_true, y_pred, 
        labels=[True, False],
    ).ravel()

    return (1 - tn / (fp + tn), tp / (tp + fn))

  def plot_specificity_sensitivity_point(clf, target_class, X_test, y_test, clf_name, ax):
      specificity, sensitivity = calc_specificity_sensitivity_point(clf, target_class, X_test, y_test)
      ax.plot(specificity, sensitivity, 'o')

  def plot_specificity_sensitivity_ci(
      clf, target_class, X, y,
      ci, clf_name,
      boot_n, ax
  ):
      (spec_low, spec_high), (sens_low, sens_high) = boot.ci(
        np.arange(len(y)),
        statfunction=lambda idx: calc_specificity_sensitivity_point(clf, target_class, X.iloc[idx], y.iloc[idx]),
        alpha = 1 - ci,
        output='lowhigh',
        n_samples=boot_n
      )

      rect = Rectangle(
          xy=(spec_low, sens_low),
          width=spec_high - spec_low,
          height=sens_high - sens_low,
          alpha=0.25,
      )
      ax.add_patch(rect)

  ```
]

For classification of objects with LDA/SVM let's filter out all classes except for:
    #calepin.chunk(
      results: "typst",
      label: "lda-svm-classes-list",
      echo: false,
    )[
      ```python
      print(pypst.Itemize([str(class_to_enlarge), str(secondary_class)]).render())
      ```
    ]
And all features except for:
    #calepin.chunk(
      results: "typst",
      label: "lda-svm-features-list",
      echo: false,
    )[
      ```python
      print(pypst.Itemize([regression_feature]).render())
      ```
    ]

#calepin.chunk(
  results: "auto",
  label: "fig-regressions",
  fig-caption: [Linear/Logistic regression],
  fig-layout-columns: (1fr, 1fr),
  fig-subcaptions: (
    [df_0],
    [df_0],
    [df_1_5],
    [df_1_5],
    [df_1_10],
    [df_1_10],
    [df_1_20],
    [df_1_20],
    [df_10_5],
    [df_10_5],
    [df_10_10],
    [df_10_10],
    [df_10_20],
    [df_10_20],
    [df_100_5],
    [df_100_5],
    [df_100_10],
    [df_100_10],
    [df_100_20],
    [df_100_20],
  ),
)[
  ```python
  for amount in dfs:
      for cord in dfs[amount]:
          data = dfs[amount][cord]
          data = data.drop(
            filter(
              lambda c: c != class_label and c != regression_feature,
              data.columns,
            ),
            axis='columns',
          )
          data = data[data[class_label].map(lambda c: c == class_to_enlarge or c == secondary_class)]
          X = data[[regression_feature]]
          y = data[class_label]

          X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=random_state,
            test_size=test_size,
          )

          linr = LinearRegression().fit(X_train, y_train)
          logr = LogisticRegression().fit(X_train, y_train)

          fig, ax = plt.subplots()

          colors = y.map(lambda q: class_to_color_map[q])
          markers = y.map(lambda q: class_to_marker_map[q])
          mscatter(X, y, ax=ax,  c=colors, m=markers)

          left, right = ax.get_xlim()
          r_x = pd.DataFrame({
              regression_feature: np.linspace(left, right, num=1024)
          })
          linr_y = linr.predict(r_x)
          logr_y = abs(class_to_enlarge - secondary_class) * logr.predict_proba(r_x)[:, 1] + min(class_to_enlarge, secondary_class)
          ax.plot(r_x, linr_y, color=(0.0, 0.0, 0.6), label="LinReg")
          ax.plot(r_x, logr_y, color=(0.0, 0.0, 0.0), label="LogReg")
          ax.legend(loc='upper right', shadow=True, fontsize='x-large')

          plt.xlabel(regression_feature)
          plt.ylabel(class_label)

          plt.show()

          fig, ax = plt.subplots()
          plot_classifier_roc_auc(logr, secondary_class, X_test, y_test, "LogReg", ax)
          plot_specificity_sensitivity_point(logr, secondary_class, X_test, y_test, "LogReg", ax)
          plot_specificity_sensitivity_ci(
              logr, secondary_class, X_test, y_test,
              confidence_level, "LogReg",
              boot_n, ax
          )

          plt.show()
  ```
]
