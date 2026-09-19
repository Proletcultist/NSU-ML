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

```python
#| results: hide
#| echo: false
import csv
from matplotlib import pyplot as plt 

with open('winequality-red.csv', newline='') as csvfile:
  csv_reader = csv.DictReader(csvfile, delimiter=',')

  data = list(csv_reader)
  fieldnames = csv_reader.fieldnames
```

#align(center, title())

= Dataset content

```python
#| results: hide
#| echo: false

total_objects_amount = len(data)
```

- Total objects amount: #py[`len(data)`]
- Total features amount: #py[`len(fieldnames) - 1`] // Since one of the fields is class, 
                                                    // substract one
// - Total classes amount: #py[`map()`]

#calepin.chunk()[
  ```python
  print(data[0])
  ```
]
