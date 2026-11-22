"""
SYNTHETIC practice questions with difficulty labels, for tests and the demo.

They are generated from templates so results are repeatable. They are NOT real
course data and say nothing about how well the model works on your course.
"""

TOPICS = [
    "binary search tree", "AVL tree", "quick sort", "merge sort",
    "Dijkstra's algorithm", "hash table", "linked list", "binary heap",
    "graph traversal", "stack", "queue", "insertion sort",
]

# Difficulty is NOT decided by the command verb alone: some "Explain" questions
# are hard (long, several parts, numbers) and some "Calculate" questions are easy.
TEMPLATES = {
    "easy": [
        "Define {t}.",
        "What is a {t}?",
        "State one property of a {t}.",
        "Explain what a {t} is.",
        "Calculate the size of a {t} that holds 3 items.",
        "List two uses of a {t}.",
        "Name the main operations of a {t}.",
    ],
    "medium": [
        "Explain how a {t} works with a simple example.",
        "Describe the steps involved in using a {t}.",
        "Differentiate between a {t} and a {u}.",
        "Define a {t} and give one example of where it is used in a real program.",
        "Calculate the number of comparisons a {t} needs for n = 8.",
        "Illustrate the {t} with a labelled diagram.",
        "Discuss the advantages of a {t} in real programs.",
    ],
    "hard": [
        "Compare a {t} and a {u} in terms of time and space complexity, and justify which is better when n = 1000000.",
        "Derive the worst case running time of a {t} and prove your answer by induction, (a) for n = 8 and (b) for general n.",
        "Explain why a {t} is preferred over a {u} in memory constrained systems, giving (a) a time argument, (b) a space argument and (c) a worked example for n = 100000.",
        "Evaluate a {t} against a {u}: calculate the number of operations for n = 1024 and justify the difference.",
        "Prove that the {t} is correct, analyse its complexity and propose an optimisation (a) in time (b) in space.",
        "Describe how a {t} behaves on sorted input, then design an improved version and evaluate its cost for n = 10000.",
        "Solve (a) insertion into, (b) deletion from and (c) search in a {t} of 15 nodes, and compare the costs.",
    ],
}


def make_questions(per_level=14):
    """[(text, difficulty), ...] in a fixed, repeatable order."""

    questions = []

    for level, templates in TEMPLATES.items():
        for i in range(per_level):
            template = templates[i % len(templates)]
            topic = TOPICS[i % len(TOPICS)]
            other = TOPICS[(i + 3) % len(TOPICS)]
            questions.append((template.format(t=topic, u=other), level))

    return questions


# Past-paper style questions the model has never seen.
PAPER_QUESTIONS = {
    "easy": ("Define a priority queue.", 2),
    "medium": ("Explain how a red-black tree stays balanced with a small example.", 5),
    "hard": ("Compare a trie and a hash table in terms of time and space complexity, and justify "
             "which is better for n = 500000 strings; derive both bounds.", 10),
}
