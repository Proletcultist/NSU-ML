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
  from scipy import stats
  from sklearn.base import clone
  from sklearn.model_selection import StratifiedKFold, KFold
  import numpy as np
  from sklearn_prg.metrics import precision_recall_gain_curve
  from sklearn.metrics import auc, roc_curve, RocCurveDisplay
  from sklearn.metrics import precision_recall_curve, auc, PrecisionRecallDisplay
  from sklearn.model_selection import train_test_split
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
  test_size = 0.25

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

  confidence_level = 0.95

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
  lda_tests = {}

  for k,df in dfs.items():
    X = df[list(lda_features)]
    y = df[class_label]

    X_train, X_test, y_train, y_test = train_test_split(
      X, y, random_state=random_state,
      test_size=test_size,
    )

    lda_clfs[k] = LinearDiscriminantAnalysis().fit(X_train, y_train)
    lda_tests[k] = (X_test, y_test)

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

= Means

#calepin.chunk(
  results: "typst",
  label: "fig-means",
  warning: false,
)[
  ```python
  table = []

  for k,df in dfs.items():
    X = df[list(lda_features)].values
    y = df[class_label].values

    clf = lda_clfs[k]

    mu0 = clf.means_[0]
    mu1 = clf.means_[1]

    midpoint = (mu0 + mu1) / 2.0

    overall_mean = X.mean(axis=0)

    d0 = clf.decision_function([mu0])[0]
    d1 = clf.decision_function([mu1])[0]
    t = d0 / (d0 - d1)
    intersection_pt = (1 - t) * mu0 + t * mu1

    table.append({
      'dataset': 'df' + str(k),
      'means midpoint': (midpoint[0] ,midpoint[1]),
      'overall mean': (overall_mean[0], overall_mean[1]),
      'intersection': (intersection_pt[0], intersection_pt[1]),
    })

  means_result = pd.DataFrame(table)

  means_result_for_display = means_result.copy()
  means_result_for_display[['means midpoint', 'overall mean', 'intersection']] = means_result_for_display[['means midpoint', 'overall mean', 'intersection']].map(lambda p: f'({sigfig.round(p[0], sigfigs=4, warn=False)}, {sigfig.round(p[1], sigfigs=4, warn=False)})')

  display_dataframe(
    means_result_for_display,
    caption='[aboba]',
  )
  ```
]

#calepin.chunk(
  results: "auto",
  label: "fig-mean-plots",
  fig-layout-columns: (1fr, 1fr),
  fig-subcaptions: (
    [X coordinate],
    [Y coordinate],
  )
)[
  ```python
  fig, ax = plt.subplots()

  ax.plot(means_result['dataset'], means_result['means midpoint'].map(lambda p: p[0]), marker='o', label='means midpoint (x)')
  ax.plot(means_result['dataset'], means_result['overall mean'].map(lambda p: p[0]), marker='s', label='overall mean (x)')
  ax.plot(means_result['dataset'], means_result['intersection'].map(lambda p: p[0]), marker='^', label='intersection point (x)')
  ax.set_xscale('log')
  ax.set_xlabel('Multiplier')
  ax.set_ylabel(f'Coordinate value ({lda_features[0]})')
  ax.grid()
  ax.legend()

  plt.show()

  fig, ax = plt.subplots()

  ax.plot(means_result['dataset'], means_result['means midpoint'].map(lambda p: p[1]), marker='o', label='means midpoint (y)')
  ax.plot(means_result['dataset'], means_result['overall mean'].map(lambda p: p[1]), marker='s', label='overall mean (y)')
  ax.plot(means_result['dataset'], means_result['intersection'].map(lambda p: p[1]), marker='^', label='intersection point (y)')
  ax.set_xscale('log')
  ax.set_xlabel('Multiplier')
  ax.set_ylabel(f'Coordinate value ({lda_features[1]})')
  ax.grid()
  ax.legend()

  plt.show()
  ```
]

= ROC, PR, PRgain

#calepin.chunk(
  results: "hide",
  label: "roc-pr-prgain-functions",
)[
  ```python
  def display_classifier_roc_auc(clf, target_class, X_test, y_test, clf_name):
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
      display.plot()

      plt.show()

  def display_classifier_pr_auc(clf, target_class, X_test, y_test, clf_name):
      precision, recall, thresholds = precision_recall_curve(
          y_test,
          clf.predict_proba(X_test)[:, np.where(clf.classes_ == target_class)[0]],
          pos_label=target_class,
      )
      
      pr_auc = auc(recall, precision)
      
      display = PrecisionRecallDisplay(
          precision=precision,
          recall=recall,
          average_precision=pr_auc,
          name=clf_name,
          pos_label=target_class,
      )
      display.plot()

      plt.show()

  def display_classifier_prgain_auc(clf, target_class, X_test, y_test, clf_name):
      y_score = clf.predict_proba(X_test)[:, np.where(clf.classes_ == target_class)[0]]
      y_true_binary = (y_test == target_class).astype(int)
      
      precision, recall, _ = precision_recall_curve(y_true_binary, y_score, pos_label=1)
      
      pi = np.mean(y_true_binary)
      
      with np.errstate(divide='ignore', invalid='ignore'):
          precision_gain = np.clip((precision - pi) / ((1 - pi) * precision), 0, 1)
          recall_gain = np.clip((recall - pi) / ((1 - pi) * recall), 0, 1)
      
      sort_idx = np.argsort(recall_gain)
      rg_sorted = recall_gain[sort_idx]
      pg_sorted = precision_gain[sort_idx]
      
      auprg = auc(rg_sorted, pg_sorted)
      
      fig, ax = plt.subplots(figsize=(6, 5))
      ax.plot(rg_sorted, pg_sorted, label=f'{clf_name} (AUPRG = {auprg:.3f})', color='tab:blue')
      
      ax.set_xlim([0.0, 1.0])
      ax.set_ylim([0.0, 1.05])
      ax.set_xlabel('Recall Gain')
      ax.set_ylabel('Precision Gain')
      ax.set_title(f'PR-Gain Curve (Target Class: {target_class})')
      ax.grid()
      ax.legend(loc="lower left")
      
      plt.show()
  ```
]

#calepin.chunk(
  results: "auto",
  label: "fig-roc-majority-class",
  fig-caption: [LDA ROC with majority class as target class],
  fig-layout-columns: (1fr, 1fr, 1fr),
)[
  ```python
  for k in dfs.keys():
    display_classifier_roc_auc(lda_clfs[k], class_to_multiply, lda_tests[k][0], lda_tests[k][1], "df" + str(k))
  ```
]

#calepin.chunk(
  results: "auto",
  label: "fig-pr-majority-class",
  fig-caption: [LDA PRC with majority class as target class],
  fig-layout-columns: (1fr, 1fr, 1fr),
)[
  ```python
  for k in dfs.keys():
    display_classifier_pr_auc(lda_clfs[k], class_to_multiply, lda_tests[k][0], lda_tests[k][1], "df" + str(k))
  ```
]

#calepin.chunk(
  results: "auto",
  label: "fig-prgain-majority-class",
  fig-caption: [LDA PRgain with majority class as target class],
  fig-layout-columns: (1fr, 1fr, 1fr),
)[
  ```python
  for k in dfs.keys():
    display_classifier_prgain_auc(lda_clfs[k], class_to_multiply, lda_tests[k][0], lda_tests[k][1], "df" + str(k))
  ```
]

#calepin.chunk(
  results: "auto",
  label: "fig-roc-minority-class",
  fig-caption: [LDA ROC with minority class as target class],
  fig-layout-columns: (1fr, 1fr, 1fr),
)[
  ```python
  for k in dfs.keys():
    display_classifier_roc_auc(lda_clfs[k], secondary_class, lda_tests[k][0], lda_tests[k][1], "df" + str(k))
  ```
]

#calepin.chunk(
  results: "auto",
  label: "fig-pr-minority-class",
  fig-caption: [LDA PRC with minority class as target class],
  fig-layout-columns: (1fr, 1fr, 1fr),
)[
  ```python
  for k in dfs.keys():
    display_classifier_pr_auc(lda_clfs[k], secondary_class, lda_tests[k][0], lda_tests[k][1], "df" + str(k))
  ```
]

#calepin.chunk(
  results: "auto",
  label: "fig-prgain-minority-class",
  fig-caption: [LDA PRgain with minority class as target class],
  fig-layout-columns: (1fr, 1fr, 1fr),
)[
  ```python
  for k in dfs.keys():
    display_classifier_prgain_auc(lda_clfs[k], secondary_class, lda_tests[k][0], lda_tests[k][1], "df" + str(k))
  ```
]

= Cross validation

#calepin.chunk(
  results: "hide",
  label: "cross-validation-functions",
)[
  ```python
  def display_roc_cross_validated(
      estimator, X, y, cv, pos_label, estimator_name, ci=0.95
  ):
      n_splits = cv.get_n_splits(X, y)
      
      y_real = []
      y_proba = []
      fold_aurocs = []

      X_arr = X.values
      y_arr = y.values
      
      for train_idx, test_idx in cv.split(X, y):
          X_train, X_test = X_arr[train_idx], X_arr[test_idx]
          y_train, y_test = y_arr[train_idx], y_arr[test_idx]
          
          estimator.fit(X_train, y_train)
          
          pred_proba = estimator.predict_proba(X_test)[:, np.where(estimator.classes_ == pos_label)[0]]
          
          fpr_fold, tpr_fold, _ = roc_curve(y_test, pred_proba, pos_label=pos_label)
          fold_aurocs.append(auc(fpr_fold, tpr_fold))
          
          y_real.append(y_test)
          y_proba.append(pred_proba)
          
      y_real_concat = np.concatenate(y_real)
      y_proba_concat = np.concatenate(y_proba)
      
      fpr_overall, tpr_overall, _ = roc_curve(y_real_concat, y_proba_concat, pos_label=pos_label)
      auc_overall = auc(fpr_overall, tpr_overall)
      
      mean_auc = np.mean(fold_aurocs)
      std_auc = np.std(fold_aurocs, ddof=1) if n_splits > 1 else 0.0
      sem_auc = std_auc / np.sqrt(n_splits) if n_splits > 1 else 0.0
      
      t_crit = stats.t.ppf((1 + ci) / 2, df=n_splits - 1) if n_splits > 1 else 1.96
      margin_error = t_crit * sem_auc
      ci_lower = np.clip(mean_auc - margin_error, 0, 1)
      ci_upper = np.clip(mean_auc + margin_error, 0, 1)
      
      fig, ax = plt.subplots(figsize=(6, 5))
      display = RocCurveDisplay(
          fpr=fpr_overall,
          tpr=tpr_overall,
          roc_auc=auc_overall,
          name=estimator_name,
          pos_label=pos_label,
      )
      display.plot(ax=ax)
      
      ax.set_title(f"Overall CV ROC Curve ({estimator_name})")
      ax.grid()
      handles, labels = ax.get_legend_handles_labels()
      labels[0] += f"\n({int(ci*100)}% CI: [{ci_lower:.3f}, {ci_upper:.3f}])"
      ax.legend(handles=handles, labels=labels, loc="lower right", fontsize="small")
      
      plt.show()

  def display_prc_cross_validated(
      estimator, X, y, cv, pos_label, estimator_name, ci=0.95
  ):
      n_splits = cv.get_n_splits(X, y)
      
      y_real = []
      y_proba = []
      fold_auprcs = []

      X_arr = X.values
      y_arr = y.values
      
      for train_idx, test_idx in cv.split(X, y):
          X_train, X_test = X_arr[train_idx], X_arr[test_idx]
          y_train, y_test = y_arr[train_idx], y_arr[test_idx]
          
          estimator.fit(X_train, y_train)
          
          pred_proba = estimator.predict_proba(X_test)[:, np.where(estimator.classes_ == pos_label)[0]]
          
          prec_fold, rec_fold, _ = precision_recall_curve(y_test, pred_proba, pos_label=pos_label)
          fold_auprcs.append(auc(rec_fold, prec_fold))
          
          y_real.append(y_test)
          y_proba.append(pred_proba)
          
      y_real_concat = np.concatenate(y_real)
      y_proba_concat = np.concatenate(y_proba)
      
      prec_overall, rec_overall, _ = precision_recall_curve(
          y_real_concat, y_proba_concat, pos_label=pos_label
      )
      auc_overall = auc(rec_overall, prec_overall)
      
      mean_auc = np.mean(fold_auprcs)
      std_auc = np.std(fold_auprcs, ddof=1) if n_splits > 1 else 0.0
      sem_auc = std_auc / np.sqrt(n_splits) if n_splits > 1 else 0.0
      
      t_crit = stats.t.ppf((1 + ci) / 2, df=n_splits - 1) if n_splits > 1 else 1.96
      margin_error = t_crit * sem_auc
      ci_lower = np.clip(mean_auc - margin_error, 0, 1)
      ci_upper = np.clip(mean_auc + margin_error, 0, 1)
      
      fig, ax = plt.subplots(figsize=(6, 5))
      display = PrecisionRecallDisplay(
          precision=prec_overall,
          recall=rec_overall,
          average_precision=auc_overall,
          name=estimator_name,
          pos_label=pos_label,
      )
      display.plot(ax=ax)
      
      ax.set_title(f"Overall CV PR Curve ({estimator_name})")
      ax.grid()
      handles, labels = ax.get_legend_handles_labels()
      labels[0] += f"\n({int(ci*100)}% CI: [{ci_lower:.3f}, {ci_upper:.3f}])"
      ax.legend(handles=handles, labels=labels, loc="lower left", fontsize="small")
      
      plt.show()
  ```
]

#calepin.chunk(
  results: "auto",
  label: "fig-prc-cross-validated",
  fig-caption: [LDA PRC with cross validation],
  fig-layout-columns: (1fr, 1fr, 1fr),
  fig-subcaptions: (
    [3-Fold],
    [5-Fold],
    [10-Fold],
    [20-Fold],
    [50-Fold],
    [100-Fold],
  )
)[
  ```python
  for folds in [3, 5, 10, 20, 50, 100]:
    X = dfs[10000][list(lda_features)]
    y = dfs[10000][class_label]

    display_prc_cross_validated(
      LinearDiscriminantAnalysis(),
      X,
      y,
      StratifiedKFold(n_splits=folds),
      secondary_class,
      "LDA",
    )
  ```
]

#calepin.chunk(
  results: "auto",
  label: "fig-roc-cross-validated",
  fig-caption: [LDA ROC with cross validation],
  fig-layout-columns: (1fr, 1fr, 1fr),
  fig-subcaptions: (
    [3-Fold],
    [5-Fold],
    [10-Fold],
    [20-Fold],
    [50-Fold],
    [100-Fold],
  )
)[
  ```python
  for folds in [3, 5, 10, 20, 50, 100]:
    X = dfs[10000][list(lda_features)]
    y = dfs[10000][class_label]

    display_roc_cross_validated(
      LinearDiscriminantAnalysis(),
      X,
      y,
      StratifiedKFold(n_splits=folds),
      secondary_class,
      "LDA",
    )
  ```
]
