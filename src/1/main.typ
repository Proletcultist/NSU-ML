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
#let py = calepin.inline.with("python")

// Imports and data reading
```python
#| results: hide
#| echo: false
import csv
import collections
import pandas as pd
import pypst
from matplotlib import pyplot as plt 

data: pd.DataFrame = pd.read_csv('data.csv')
```

#align(center, title())

= Dataset content

```python
#| results: hide
#| echo: false

objects_classes_by_classes_count = (
    data
    .groupby(['quality'])
    .size()
    .reset_index(name='Objects count')
)
```

- Total objects amount: #py[`len(data.index)`]
- Total features amount: #py[`len(data.columns) - 1`] // Since one of the fields is class, 
                                                      // substract one
- Total classes amount: #py[`len(objects_classes_by_classes_count.index)`]
- Distribution of objects into classes:

#calepin.chunk(
  echo: false,
  results: "typst",
)[
  ```python
  table = pypst.Table.from_dataframe(objects_classes_by_classes_count, include_index = False)
  figure = pypst.Figure(table, caption='[Objects count by classes]')

  print(figure.render())
  ```
]
