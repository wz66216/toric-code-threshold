from __future__ import annotations

from dataclasses import dataclass
import numpy as np

Vertex = tuple[int, int]


@dataclass(frozen=True)
class ToricChain:
    horizontal: np.ndarray
    vertical: np.ndarray

    def __post_init__(self) -> None:
        h = np.asarray(self.horizontal, dtype=bool)
        v = np.asarray(self.vertical, dtype=bool)
        if h.shape != v.shape:
            raise ValueError("horizontal and vertical arrays must have identical shape")
        if h.ndim != 2 or h.shape[0] != h.shape[1]:
            raise ValueError("chain arrays must be square two-dimensional arrays")
        object.__setattr__(self, "horizontal", h.copy())
        object.__setattr__(self, "vertical", v.copy())

    @property
    def L(self) -> int:
        return int(self.horizontal.shape[0])

    @classmethod
    def empty(cls, L: int) -> "ToricChain":
        if L <= 0:
            raise ValueError("L must be positive")
        return cls(np.zeros((L, L), dtype=bool), np.zeros((L, L), dtype=bool))

    def copy_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return self.horizontal.copy(), self.vertical.copy()

    def weight(self) -> int:
        return int(np.count_nonzero(self.horizontal) + np.count_nonzero(self.vertical))

    def with_horizontal(self, x: int, y: int) -> "ToricChain":
        h, v = self.copy_arrays()
        h[x % self.L, y % self.L] ^= True
        return ToricChain(h, v)

    def with_vertical(self, x: int, y: int) -> "ToricChain":
        h, v = self.copy_arrays()
        v[x % self.L, y % self.L] ^= True
        return ToricChain(h, v)

    def __add__(self, other: object) -> "ToricChain":
        if not isinstance(other, ToricChain):
            return NotImplemented
        if self.L != other.L:
            raise ValueError("cannot add chains with different L")
        return ToricChain(np.logical_xor(self.horizontal, other.horizontal), np.logical_xor(self.vertical, other.vertical))

    def syndrome(self) -> np.ndarray:
        h = self.horizontal
        v = self.vertical
        return np.logical_xor.reduce(
            [
                np.roll(h, 1, axis=0),
                h,
                np.roll(v, 1, axis=1),
                v,
            ]
        )

    def defects(self) -> list[Vertex]:
        xs, ys = np.where(self.syndrome())
        return [(int(x), int(y)) for x, y in zip(xs, ys)]

    def logical_parity(self) -> tuple[bool, bool]:
        x_winding = bool(np.count_nonzero(self.horizontal[self.L - 1, :]) % 2)
        y_winding = bool(np.count_nonzero(self.vertical[:, self.L - 1]) % 2)
        return x_winding, y_winding

    def is_logical_failure(self) -> bool:
        return any(self.logical_parity())


def _signed_shortest_delta(start: int, end: int, L: int) -> int:
    forward = (end - start) % L
    backward = forward - L
    if abs(forward) <= abs(backward):
        return int(forward)
    return int(backward)


def shortest_path_chain(L: int, start: Vertex, end: Vertex) -> ToricChain:
    """Construct one deterministic shortest toric path between two vertices."""
    x, y = start
    target_x, target_y = end
    chain = ToricChain.empty(L)

    dx = _signed_shortest_delta(x, target_x, L)
    step_x = 1 if dx >= 0 else -1
    for _ in range(abs(dx)):
        if step_x == 1:
            chain = chain.with_horizontal(x, y)
            x = (x + 1) % L
        else:
            x = (x - 1) % L
            chain = chain.with_horizontal(x, y)

    dy = _signed_shortest_delta(y, target_y, L)
    step_y = 1 if dy >= 0 else -1
    for _ in range(abs(dy)):
        if step_y == 1:
            chain = chain.with_vertical(x, y)
            y = (y + 1) % L
        else:
            y = (y - 1) % L
            chain = chain.with_vertical(x, y)

    return chain


def chain_from_edges(
    L: int,
    horizontal_edges: tuple[Vertex, ...] = (),
    vertical_edges: tuple[Vertex, ...] = (),
) -> ToricChain:
    chain = ToricChain.empty(L)
    for x, y in horizontal_edges:
        chain = chain.with_horizontal(x, y)
    for x, y in vertical_edges:
        chain = chain.with_vertical(x, y)
    return chain


def random_error(L: int, p: float, rng: np.random.Generator | None = None) -> ToricChain:
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    generator = np.random.default_rng() if rng is None else rng
    return ToricChain(generator.random((L, L)) < p, generator.random((L, L)) < p)
