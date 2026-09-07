#!/usr/bin/env python3
"""Plot nine traversals of a fixed, labeled 6x6 grid."""
import math
import random
import matplotlib.pyplot as plt

N, SEED = 6, 2026
BLUE, GREEN, RED, EDGE, BLACK = "#8ecae6", "#55a868", "#e76f51", "#263238", "#111111"


def fixed_label(p):
    return p[1] * N + p[0] + 1


def rotate_start(order):
    i = order.index((0, 0))
    return order[i:] + order[:i]


def hilbert_rotate(s, x, y, rx, ry):
    if ry == 0:
        if rx == 1:
            x, y = s - 1 - x, s - 1 - y
        x, y = y, x
    return x, y


def hilbert_index(x, y, side=8):
    d, s = 0, side // 2
    while s:
        rx, ry = int(bool(x & s)), int(bool(y & s))
        d += s * s * ((3 * rx) ^ ry)
        x, y = hilbert_rotate(s, x, y, rx, ry)
        s //= 2
    return d


def morton_code(x, y):
    code = 0
    for bit in range(3):
        code |= ((x >> bit) & 1) << (2 * bit)
        code |= ((y >> bit) & 1) << (2 * bit + 1)
    return code


def spiral_order():
    """Adjacent-step rectangular spiral from node 1 toward the center."""
    result = []
    left, right, bottom, top = 0, N - 1, 0, N - 1
    while left <= right and bottom <= top:
        result.extend((x, bottom) for x in range(left, right + 1))
        bottom += 1
        result.extend((right, y) for y in range(bottom, top + 1))
        right -= 1
        if bottom <= top:
            result.extend((x, top) for x in range(right, left - 1, -1))
            top -= 1
        if left <= right:
            result.extend((left, y) for y in range(top, bottom - 1, -1))
            left += 1
    return result


def build_orderings():
    points = [(x, y) for y in range(N) for x in range(N)]
    center = (N - 1) / 2

    # xy lexicographic: x is primary, y secondary.
    lexicographic = sorted(points, key=lambda p: (p[0], p[1]))

    # Paper-style vertical serpent through pairs of consecutive columns.
    snake = []
    for x in range(N):
        column = [(x, y) for y in range(N)]
        snake.extend(column if x % 2 == 0 else reversed(column))

    # Diagonal projection onto (1, 1), with deterministic ties.
    projection = sorted(points, key=lambda p: (p[0] + p[1], p[0], p[1]))

    # yx sweep: y is primary, x secondary.
    sweep = sorted(points, key=lambda p: (p[1], p[0]))

    random_order = points[1:]
    random.Random(SEED).shuffle(random_order)
    random_order.insert(0, (0, 0))

    def angular_key(p):
        dx, dy = p[0] - center, p[1] - center
        return math.atan2(dy, dx), dx * dx + dy * dy, p[1], p[0]

    angular = rotate_start(sorted(points, key=angular_key))
    hilbert = rotate_start(sorted(points, key=lambda p: hilbert_index(*p)))
    morton = rotate_start(sorted(points, key=lambda p: morton_code(*p)))

    orders = [
        ("Lexicographic (xy)", lexicographic), ("Snake Move", snake),
        ("Projection", projection), ("Sweep (yx)", sweep),
        ("Spiral", spiral_order()), ("Random", random_order),
        ("Angular", angular), ("Hilbert", hilbert), ("Morton", morton),
    ]
    expected = set(points)
    for name, order in orders:
        assert len(order) == 36 and len(set(order)) == 36
        assert set(order) == expected and order[0] == (0, 0), name
    return orders


def draw(ax, title, order):
    xs, ys = zip(*order)
    ax.plot(xs, ys, color=BLACK, linewidth=1.45, solid_capstyle="round",
            solid_joinstyle="round", zorder=1)
    colors = [GREEN if p == (0, 0) else RED if p == order[-1] else BLUE for p in order]
    ax.scatter(xs, ys, s=330, c=colors, edgecolors=EDGE, linewidths=0.75, zorder=2)
    for x, y in order:
        ax.text(x, y, str(fixed_label((x, y))), ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="#17202a", zorder=3)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set(xlim=(-0.55, 5.55), ylim=(-0.55, 5.55))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 13), facecolor="white")
    for ax, (title, order) in zip(axes.flat, build_orderings()):
        draw(ax, title, order)
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.035, top=0.965,
                        wspace=0.22, hspace=0.27)
    fig.savefig("ordering_strategies_3x3_fixedlabels.png", dpi=450,
                bbox_inches="tight", facecolor="white")
    fig.savefig("ordering_strategies_3x3_fixedlabels.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
