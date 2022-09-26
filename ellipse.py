import numpy as np

# mostly adapted from https://gist.github.com/drawable/92792f59b6ff8869d8b1


class Vec2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Ellipse:
    def __init__(self, rx, ry, rotation, origin=None):
        if origin is None:
            origin = Vec2D(0, 0)
        self.o = origin
        self.rx = rx
        self.ry = ry
        self.rotation = rotation

        self.quadratic = self.get_quadratic()

    def get_quadratic(self):
        a = self.o.x
        b = self.rx * self.rx
        c = self.o.y
        d = self.ry * self.ry
        A = np.cos(-self.rotation)
        B = np.sin(-self.rotation)

        return [
            A * A / b + B * B / d,
            2 * A * B / d - 2 * A * B / b,
            A * A / d + B * B / b,
            (2 * A * B * c - 2 * a * A * A) / b + (-2 * a * B * B - 2 * A * B * c) / d,
            (2 * a * A * B - 2 * B * B * c) / b + (-2 * a * A * B - 2 * A * A * c) / d,
            (a * a * A * A - 2 * a * A * B * c + B * B * c * c) / b
            + (a * a * B * B + 2 * a * A * B * c + A * A * c * c) / d
            - 1,
        ]

    def plot(self, ax):
        t = np.linspace(0, 2 * np.pi, 100)

        ell = np.array([self.o.x + self.rx * np.cos(t), self.o.y + self.ry * np.sin(t)])
        rot = np.array(
            [
                [np.cos(self.rotation), -np.sin(self.rotation)],
                [np.sin(self.rotation), np.cos(self.rotation)],
            ]
        )
        ell_rotated = np.dot(rot, ell)

        ax.plot(ell_rotated[0], ell_rotated[1])
        return ax


def quartics(el1, el2):
    a1, b1, c1, d1, e1, f1 = el1.quadratic
    a2, b2, c2, d2, e2, f2 = el2.quadratic

    return [
        f1 * a1 * d2 * d2
        + a1 * a1 * f2 * f2
        - d1 * a1 * d2 * f2
        + a2 * a2 * f1 * f1
        - 2 * a1 * f2 * a2 * f1
        - d1 * d2 * a2 * f1
        + a2 * d1 * d1 * f2,
        e2 * d1 * d1 * a2
        - f2 * d2 * a1 * b1
        - 2 * a1 * f2 * a2 * e1
        - f1 * a2 * b2 * d1
        + 2 * d2 * b2 * a1 * f1
        + 2 * e2 * f2 * a1 * a1
        + d2 * d2 * a1 * e1
        - e2 * d2 * a1 * d1
        - 2 * a1 * e2 * a2 * f1
        - f1 * a2 * d2 * b1
        + 2 * f1 * e1 * a2 * a2
        - f2 * b2 * a1 * d1
        - e1 * a2 * d2 * d1
        + 2 * f2 * b1 * a2 * d1,
        e2 * e2 * a1 * a1
        + 2 * c2 * f2 * a1 * a1
        - e1 * a2 * d2 * b1
        + f2 * a2 * b1 * b1
        - e1 * a2 * b2 * d1
        - f2 * b2 * a1 * b1
        - 2 * a1 * e2 * a2 * e1
        + 2 * d2 * b2 * a1 * e1
        - c2 * d2 * a1 * d1
        - 2 * a1 * c2 * a2 * f1
        + b2 * b2 * a1 * f1
        + 2 * e2 * b1 * a2 * d1
        + e1 * e1 * a2 * a2
        - c1 * a2 * d2 * d1
        - e2 * b2 * a1 * d1
        + 2 * f1 * c1 * a2 * a2
        - f1 * a2 * b2 * b1
        + c2 * d1 * d1 * a2
        + d2 * d2 * a1 * c1
        - e2 * d2 * a1 * b1
        - 2 * a1 * f2 * a2 * c1,
        -2 * a1 * a2 * c1 * e2
        + e2 * a2 * b1 * b1
        + 2 * c2 * b1 * a2 * d1
        - c1 * a2 * b2 * d1
        + b2 * b2 * a1 * e1
        - e2 * b2 * a1 * b1
        - 2 * a1 * c2 * a2 * e1
        - e1 * a2 * b2 * b1
        - c2 * b2 * a1 * d1
        + 2 * e2 * c2 * a1 * a1
        + 2 * e1 * c1 * a2 * a2
        - c1 * a2 * d2 * b1
        + 2 * d2 * b2 * a1 * c1
        - c2 * d2 * a1 * b1,
        a1 * a1 * c2 * c2
        - 2 * a1 * c2 * a2 * c1
        + a2 * a2 * c1 * c1
        - b1 * a1 * b2 * c2
        - b1 * b2 * a2 * c1
        + b1 * b1 * a2 * c2
        + c1 * a1 * b2 * b2,
    ]


def getY(quartics):
    """Calculates the rational roots of the quartics

    Args:
        quartics (_type_): _description_

    Returns:
        _type_: _description_
    """
    e, d, c, b, a = quartics

    delta = (
        256 * a * a * a * e * e * e
        - 192 * a * a * b * d * e * e
        - 128 * a * a * c * c * e * e
        + 144 * a * a * c * d * d * e
        - 27 * a * a * d * d * d * d
        + 144 * a * b * b * c * e * e
        - 6 * a * b * b * d * d * e
        - 80 * a * b * c * c * d * e
        + 18 * a * b * c * d * d * d
        + 16 * a * c * c * c * c * e
        - 4 * a * c * c * c * d * d
        - 27 * b * b * b * b * e * e
        + 18 * b * b * b * c * d * e
        - 4 * b * b * b * d * d * d
        - 4 * b * b * c * c * c * e
        + b * b * c * c * d * d
    )

    P = 8 * a * c - 3 * b * b
    D = (
        64 * a * a * a * e
        - 16 * a * a * c * c
        + 16 * a * b * b * c
        - 16 * a * a * b * d
        - 3 * b * b * b * b
    )

    d0 = c * c - 3 * b * d + 12 * a * e
    d1 = (
        2 * c * c * c - 9 * b * c * d + 27 * b * b * e + 27 * a * d * d - 72 * a * c * e
    )

    p = (8 * a * c - 3 * b * b) / (8 * a * a)
    q = (b * b * b - 4 * a * b * c + 8 * a * a * d) / (8 * a * a * a)

    phi = np.arccos(d1 / (2 * np.sqrt(d0 * d0 * d0)))

    if np.isnan(phi):  # & (d1 == 0):
        Q = d1 + np.sqrt(d1 * d1 - 4 * d0 * d0 * d0)
        Q = Q / 2
        Q = np.power(Q, 1 / 3)
        S = 0.5 * np.sqrt(-2 / 3 * p + (1 / (3 * a)) * (Q + d0 / Q))
    else:
        S = 0.5 * np.sqrt(-2 / 3 * p + 2 / (3 * a) * np.sqrt(d0) * np.cos(phi / 3))

    y = []
    R = -4 * S * S - 2 * p + q / S

    if R > 0:
        R = 0.5 * np.sqrt(R)
        y.append(-b / (4 * a) - S + R)
        y.append(-b / (4 * a) - S - R)
    else:
        y.append(-b / (4 * a) - S)

    R = -4 * S * S - 2 * p - q / S
    if R > 0:
        R = 0.5 * np.sqrt(R)
        y.append(-b / (4 * a) + S + R)
        y.append(-b / (4 * a) + S - R)
    else:
        y.append(-b / (4 * a) + S)
    return np.array(y)


def getX(y, el1, el2):
    a1, b1, c1, d1, e1, f1 = el1.quadratic
    a2, b2, c2, d2, e2, f2 = el2.quadratic

    if len(y) == 2:
        bb = b1 * y + d1
        cc = c1 * y**2 + e1 * y + f1
        x = (-bb + np.sqrt(bb**2 - 4 * a1 * cc)) / (2 * a1)
        return x
    else:
        x = -(
            a1 * f2
            + a1 * c2 * y * y
            - a2 * c1 * y * y
            + a1 * e2 * y
            - a2 * e1 * y
            - a2 * f1
        ) / (a1 * b2 * y + a1 * d2 - a2 * b1 * y - a2 * d1)

    return np.array(x)


def intersection(el1, el2):
    """Calculates the intersection between two ellipses
    THIS DESPERATELY NEEDS CHECKING. ONLY WORKS WHEN THEY ARE ACTUALLY INTERSECTING. LOL

    Args:
        el1 (_type_): _description_
        el2 (_type_): _description_
    """
    q = quartics(el1, el2)
    y = getY(q)
    x = getX(y, el1, el2)
    return x, y
