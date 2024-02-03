import numpy as np
from collections import Counter


def clamp(i, left, right):
    return min(max(i, left), right)


def grid_interp(x: float, fp: np.ndarray, x_min: float, dx: float):
    r"""
    For function :math:`f` defined by f(x_min + i*dx) = fp[i], compute
    f(x) by linear interpolation.
    """

    x_max = x_min + (len(fp) - 1) * dx
    x_clamp = clamp(x, x_min, x_max)
    x_left_idx = clamp(int(np.floor((x_clamp - x_min) / dx)), 0, len(fp) - 1)
    x_right_idx = clamp(int(np.ceil((x_clamp - x_min) / dx)), 0, len(fp) - 1)

    return np.interp(
        x_clamp,
        [x_min + dx * x_left_idx, x_min + dx * x_right_idx],
        fp[[x_left_idx, x_right_idx]],
    )


def tree_size(tree):
    num_events = 0

    for _ in tree.traverse("preorder"):
        num_events += 1

    return num_events


def num_subtrees_by_size(tree):
    size_by_node = {}

    for node in tree.traverse("postorder"):
        if node.is_leaf():
            size_by_node[node.name] = 1
        else:
            size_by_node[node.name] = sum(
                [size_by_node[child.name] for child in node.children]
            )

    subtree_counts = Counter(size_by_node.values())
    return subtree_counts


def total_branch_length(tree):
    return {"total": sum([node.dist for node in tree.traverse() if not node.is_root()])}


def num_lineages_by_phenotype_and_time(tree, t):
    return Counter(
        [
            node.state
            for node in tree.iter_descendants()
            if node.t >= t and node.up.t <= t
        ]
    )


def num_nodes_by_phenotype(tree):
    return Counter([node.state for node in tree.traverse()])


def num_leaves_by_phenotype(tree):
    return Counter([node.state for node in tree.traverse() if node.is_leaf()])


def branch_length_by_phenotype(tree):
    branch_lengths = {}

    for node in tree.iter_descendants():
        if node.up.state not in branch_lengths:
            branch_lengths[node.up.state] = 0.0
        branch_lengths[node.up.state] += node.dist

    return branch_lengths


def clade_sizes_by_phenotype(tree):
    clade_sizes = []

    subclade_size_by_node = {}
    for node in tree.traverse("postorder"):
        subclade_size_by_node[node.name] = 1 + sum(
            [
                subclade_size_by_node[child.name]
                for child in node.children
                if child.state == node.state
            ]
        )
        if (
            node.up is None or node.up.state != node.state
        ):  # in this case, node is the root of a subclade
            clade_sizes.append(
                {"phenotype": node.state, "value": subclade_size_by_node[node.name]}
            )

    return clade_sizes


def clade_lengths_by_phenotype(tree):
    clade_lengths = []

    subclade_length_by_node = {}
    for node in tree.traverse("postorder"):
        subclade_length_by_node[node.name] = sum(
            [child.dist for child in node.children]
        ) + sum(
            [
                subclade_length_by_node[child.name]
                for child in node.children
                if child.state == node.state
            ]
        )  # branch associate with parent phenotype
        if (
            node.up is None or node.up.state != node.state
        ):  # in this case, node is the root of a subclade
            clade_lengths.append(
                {"phenotype": node.state, "value": subclade_length_by_node[node.name]}
            )

    return clade_lengths


def forward_moment_equation(
    state_space,
    birth_rates: np.ndarray,
    death_rates: np.ndarray,
    mutation_rates: np.ndarray,
    transition_matrix: np.ndarray,
    t_min: float,
    t_max: float,
    dt: float,
):
    pass
