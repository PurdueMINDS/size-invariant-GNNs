import torch

graphlet = {
    3: {
        "Wedge": [
            [1, 2],
            [2, 1],
            [2, 3],
            [3, 2],
        ],

        "Triangle": [
            [1, 2],
            [1, 3],
            [2, 1],
            [2, 3],
            [3, 1],
            [3, 2],
        ]
    },

    4: {
        "3-star": [
            [1, 3],
            [2, 3],
            [3, 1],
            [3, 2],
            [3, 4],
            [4, 3],
        ],

        "3-path": [
            [1, 2],
            [1, 3],
            [2, 1],
            [3, 1],
            [3, 4],
            [4, 3],
        ],

        "Tailed triangle": [
            [1, 2],
            [1, 3],
            [2, 1],
            [2, 3],
            [3, 1],
            [3, 2],
            [3, 4],
            [4, 3],
        ],

        "4-cycle": [
            [1, 2],
            [1, 3],
            [2, 1],
            [2, 4],
            [3, 1],
            [3, 4],
            [4, 2],
            [4, 3],
        ],

        "Diamond": [
            [1, 2],
            [1, 3],
            [2, 1],
            [2, 3],
            [2, 4],
            [3, 1],
            [3, 2],
            [3, 4],
            [4, 2],
            [4, 3]
        ],

        "4-clique": [
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 1],
            [2, 3],
            [2, 4],
            [3, 1],
            [3, 2],
            [3, 4],
            [4, 1],
            [4, 2],
            [4, 3],
        ]
    },

    5 : {
        "4-star" : [
            [1, 2],
            [1, 3],
            [1, 4],
            [1, 5],
            [2, 1],
            [3, 1],
            [4, 1],
            [5, 1],
        ],

        "Prong": [
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 1],
            [2, 5],
            [3, 1],
            [4, 1],
            [5, 2],
        ],

        "4-path": [
            [1, 2],
            [1, 3],
            [2, 1],
            [2, 4],
            [3, 1],
            [3, 5],
            [4, 2],
            [5, 3],
        ],

        "Forktailed-tri": [
            [1, 2],
            [1, 3],
            [2, 1],
            [2, 3],
            [3, 1],
            [3, 2],
            [3, 4],
            [3, 5],
            [4, 3],
            [5, 3],
        ],

        "Lontailed-tri": [
            [1, 2],
            [1, 3],
            [2, 1],
            [2, 3],
            [2, 4],
            [3, 1],
            [3, 2],
            [4, 2],
            [4, 5],
            [5, 4],
        ],

        "Doubletailed-tri": [
            [1, 2],
            [1, 3],
            [2, 1],
            [2, 3],
            [2, 4],
            [3, 1],
            [3, 2],
            [3, 5],
            [4, 2],
            [5, 3],
        ],

        "Tailed-4-cycle": [
            [1, 2],
            [1, 3],
            [2, 1],
            [2, 4],
            [3, 1],
            [3, 4],
            [3, 5],
            [4, 2],
            [4, 3],
            [5, 3],
        ],

        "5-cycle": [
            [1, 2],
            [1, 3],
            [2, 1],
            [2, 4],
            [3, 1],
            [3, 5],
            [4, 2],
            [4, 5],
            [5, 3],
            [5, 4],
        ],

        "Hourglass": [
            [1, 2],
            [1, 3],
            [2, 1],
            [2, 3],
            [3, 1],
            [3, 2],
            [3, 4],
            [3, 5],
            [4, 3],
            [4, 5],
            [5, 3],
            [5, 4],
        ],

        "Cobra": [
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 1],
            [2, 4],
            [3, 1],
            [3, 4],
            [3, 5],
            [4, 1],
            [4, 2],
            [4, 3],
            [5, 3],
        ],

        "Stingray": [
            [1, 2],
            [1, 3],
            [2, 1],
            [2, 3],
            [2, 4],
            [3, 1],
            [3, 2],
            [3, 4],
            [3, 5],
            [4, 2],
            [4, 3],
            [5, 3],
        ],

        "Hatted-4-cycle": [
            [1, 2],
            [1, 3],
            [2, 1],
            [2, 4],
            [3, 1],
            [3, 4],
            [3, 5],
            [4, 2],
            [4, 3],
            [4, 5],
            [5, 3],
            [5, 4],
        ],

        "3-wedge-col": [
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 1],
            [2, 5],
            [3, 1],
            [3, 5],
            [4, 1],
            [4, 5],
            [5, 2],
            [5, 3],
            [5, 4],
        ],

        "3-tri-collision": [
            [1, 2],
            [1, 3],
            [1, 4],
            [1, 5],
            [2, 1],
            [2, 5],
            [3, 1],
            [3, 5],
            [4, 1],
            [4, 5],
            [5, 1],
            [5, 2],
            [5, 3],
            [5, 4],
        ],

        "Tailed-4-clique": [
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 1],
            [2, 3],
            [2, 4],
            [3, 1],
            [3, 2],
            [3, 4],
            [3, 5],
            [4, 1],
            [4, 2],
            [4, 3],
            [5, 3],
        ],

        "Triangle-strip": [
            [1, 2],
            [1, 3],
            [2, 1],
            [2, 3],
            [2, 4],
            [3, 1],
            [3, 2],
            [3, 4],
            [3, 5],
            [4, 2],
            [4, 3],
            [4, 5],
            [5, 3],
            [5, 4],
        ],

        "Diamond-wed-col": [
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 1],
            [2, 3],
            [2, 4],
            [3, 1],
            [3, 2],
            [3, 5],
            [4, 1],
            [4, 2],
            [4, 5],
            [5, 3],
            [5, 4],
        ],

        "4-wheel" : [
            [1, 2],
            [1, 3],
            [1, 4],
            [1, 5],
            [2, 1],
            [2, 3],
            [2, 4],
            [3, 1],
            [3, 2],
            [3, 5],
            [4, 1],
            [4, 2],
            [4, 5],
            [5, 1],
            [5, 3],
            [5, 4],
        ],

        "Hatted-4-clique": [
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 1],
            [2, 3],
            [2, 4],
            [3, 1],
            [3, 2],
            [3, 4],
            [3, 5],
            [4, 1],
            [4, 2],
            [4, 3],
            [4, 5],
            [5, 3],
            [5, 4],
        ],

        "Almost-5-clique": [
            [1, 3],
            [1, 4],
            [1, 5],
            [2, 3],
            [2, 4],
            [2, 5],
            [3, 1],
            [3, 2],
            [3, 4],
            [3, 5],
            [4, 1],
            [4, 2],
            [4, 3],
            [4, 5],
            [5, 1],
            [5, 2],
            [5, 3],
            [5, 4],
        ],

        "5-clique": [
            [1, 2],
            [1, 3],
            [1, 4],
            [1, 5],
            [2, 1],
            [2, 3],
            [2, 4],
            [2, 5],
            [3, 1],
            [3, 2],
            [3, 4],
            [3, 5],
            [4, 1],
            [4, 2],
            [4, 3],
            [4, 5],
            [5, 1],
            [5, 2],
            [5, 3],
            [5, 4],
        ],

    }
}

graphlets_tensor = {
    k: {
        name: torch.tensor(graphlet[k][name]).permute(1, 0) for name in graphlet[k]
    }
    for k in graphlet
}

if __name__ == "__main__":
    import networkx as nx
    import matplotlib.pyplot as plt

    assert all([
        [u, v] in graphlet[size][name]
        for size in graphlet
            for name in graphlet[size]
                for v, u in graphlet[size][name]
    ]), "Some reverse edges are missing"

    for size in graphlet:
        for name in graphlet[size]:
            G = nx.Graph()
            G.add_edges_from(graphlet[size][name])
            nx.draw(G)
            plt.show()