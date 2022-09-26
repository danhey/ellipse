Fast analytic solver for the intersection of two ellipses of arbitrary rotation in Python
## Usage

Basic syntax of creating an ellipse:
```python
Ellipse(rx, ry, rotation, origin)
```

To calculate the intersection of two ellipses
```python
from ellipse import Ellipse, intersection
el1 = Ellipse(5, 20, 1)
el2 = Ellipse(5, 10, 0)
x, y = intersection(el1, el2)
```

To plot the intersection
```python
fig, ax = plt.subplots()

el1.plot(ax)
el2.plot(ax)

ax.scatter(x, y)
```

![example](/docs/images/example.png)
