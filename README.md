Fast analytic solver for the intersection of two ellipses of arbitrary rotation in Python
## Usage

```python
from ellipse import Ellipse, intersection
el1 = Ellipse(5,20,1)
el2 = Ellipse(5,10,0)
x, y = intersection(el1, el2)
```

![example][docs/imagesexample.png]