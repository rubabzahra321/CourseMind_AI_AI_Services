"""A small, realistic Data Structures question set for the ML tests."""

CORPUS = {
    "Trees": [
        "Define a binary search tree and state its properties.",
        "Explain inorder, preorder and postorder tree traversal.",
        "What is the height of a balanced binary tree with n nodes?",
        "Insert the given keys into an AVL tree and show the rotations.",
        "Describe a heap and a complete binary tree.",
        "Compare the search complexity of a BST and an AVL tree.",
        "Delete a node with two children from a binary search tree.",
        "What is a leaf node and the root of a tree?",
    ],
    "Graphs": [
        "Explain breadth first search on a graph with an example.",
        "Describe Dijkstra's shortest path algorithm.",
        "What is a minimum spanning tree? Explain Kruskal's algorithm.",
        "Represent a graph using an adjacency list and an adjacency matrix.",
        "How do you detect a cycle in a directed graph?",
        "Explain topological sort of a directed acyclic graph.",
        "Differentiate between directed and undirected graphs.",
        "Apply depth first search to find connected components of a graph.",
    ],
    "Sorting": [
        "Compare quick sort and merge sort in terms of complexity.",
        "Explain bubble sort with a worked example.",
        "Describe the divide and conquer strategy used in merge sort.",
        "What is a stable sorting algorithm? Is insertion sort stable?",
        "Analyse the worst case running time of quick sort.",
        "Explain the heap sort procedure step by step.",
        "Sort the list using selection sort and count the swaps.",
        "Why is counting sort faster than comparison based sorting?",
    ],
}

# Fresh questions the model has never seen, with the topic a reader would pick.
NEW_QUESTIONS = {
    "Graphs": "Explain how breadth first search and depth first search traverse a graph.",
    "Trees": "Insert these keys into a binary search tree and give the inorder traversal.",
    "Sorting": "Explain why quick sort has a bad worst case and how merge sort avoids it.",
}
