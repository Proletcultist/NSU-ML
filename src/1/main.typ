#import "/.calepin/calepin.typ" as calepin
#show: calepin.document

#set document(
  title: [Лабораторная работа №1 \ Зенин Матвей 24214],
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
from matplotlib import pyplot as plt 

with open('data.csv', newline='') as csvfile:
  csv_reader = csv.DictReader(csvfile, delimiter=',')

  data = list(csv_reader)
  fieldnames = csv_reader.fieldnames
```

#align(center, title())

= Dataset content

```python
#| results: hide
#| echo: false

objects_classes_count = collections.Counter(row['quality'] for row in data)
```

- Total objects amount: #py[`len(data)`]
- Total features amount: #py[`len(fieldnames) - 1`] // Since one of the fields is class, 
                                                    // substract one
- Total classes amount: #py[`len(objects_classes_count)`]

Classes: #py[`objects_classes_count`]

#calepin.chunk()[
  ```python
  print(data[0])
  ```
]
