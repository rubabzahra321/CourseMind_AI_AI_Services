import copy

from app.search.algorithms import bfs_prerequisites, dfs_topic_subtree, greedy_best_first, minimax


def edge(topic, prereq):
    return {"topic_id": topic, "prerequisite_topic_id": prereq}


# ---------------------------------------------------------------- BFS
def test_bfs_returns_chain_from_foundation_to_target():
    edges = [edge("C", "B"), edge("B", "A")]
    assert bfs_prerequisites("C", edges) == ["A", "B", "C"]


def test_bfs_topic_without_prerequisites_returns_itself():
    assert bfs_prerequisites("X", []) == ["X"]
    assert bfs_prerequisites("X", [edge("Y", "Z")]) == ["X"]


def test_bfs_terminates_on_cycle():
    edges = [edge("A", "B"), edge("B", "C"), edge("C", "A")]
    result = bfs_prerequisites("A", edges)
    assert result[-1] == "A"  # never loops forever


def test_bfs_diamond_picks_a_valid_shortest_chain():
    #   D needs B and C ; both need A
    edges = [edge("D", "B"), edge("D", "C"), edge("B", "A"), edge("C", "A")]
    path = bfs_prerequisites("D", edges)
    assert path[0] == "A" and path[-1] == "D" and len(path) == 3


# ---------------------------------------------------------------- DFS
def test_dfs_collects_whole_subtree_with_depths():
    topics = [
        {"id": "unit", "parent_topic_id": None},
        {"id": "t1", "parent_topic_id": "unit"},
        {"id": "t1a", "parent_topic_id": "t1"},
        {"id": "t2", "parent_topic_id": "unit"},
        {"id": "other", "parent_topic_id": None},
    ]
    result = dfs_topic_subtree("unit", topics)
    assert [(n["topic_id"], n["depth"]) for n in result] == [("unit", 0), ("t1", 1), ("t1a", 2), ("t2", 1)]


def test_dfs_leaf_returns_only_itself():
    assert dfs_topic_subtree("t", [{"id": "t", "parent_topic_id": None}]) == [{"topic_id": "t", "depth": 0}]


def test_dfs_is_cycle_safe():
    topics = [{"id": "a", "parent_topic_id": "b"}, {"id": "b", "parent_topic_id": "a"}]
    ids = [n["topic_id"] for n in dfs_topic_subtree("a", topics)]
    assert sorted(ids) == ["a", "b"]


# ---------------------------------------------------------------- Greedy best-first
def test_greedy_orders_by_utility_desc_and_respects_limit():
    topics = [{"id": 1, "utility_score": 10}, {"id": 2, "utility_score": 90}, {"id": 3, "utility_score": 50}]
    assert [t["id"] for t in greedy_best_first(topics)] == [2, 3, 1]
    assert [t["id"] for t in greedy_best_first(topics, limit=2)] == [2, 3]
    assert greedy_best_first([]) == []


def test_greedy_does_not_mutate_input():
    topics = [{"id": 1, "utility_score": 1}, {"id": 2, "utility_score": 2}]
    snapshot = copy.deepcopy(topics)
    greedy_best_first(topics)
    assert topics == snapshot


# ---------------------------------------------------------------- Minimax (AI = "O", student = "X")
def test_minimax_takes_immediate_win():
    board = ["O", "O", "", "X", "X", "", "", "", ""]
    score, pos = minimax(board, True)
    assert pos == 2 and score > 0


def test_minimax_blocks_student_win():
    board = ["X", "X", "", "", "O", "", "", "", ""]
    _, pos = minimax(board, True)
    assert pos == 2


def test_minimax_leaves_board_unchanged():
    board = ["X", "", "", "", "O", "", "", "", ""]
    snapshot = list(board)
    minimax(board, True)
    assert board == snapshot


def test_minimax_on_finished_board_returns_no_move():
    assert minimax(["X", "X", "X", "O", "O", "", "", "", ""], True)[1] is None
    assert minimax(["X", "O", "X", "X", "O", "O", "O", "X", "X"], True) == (0, None)


def test_ai_can_never_lose_against_any_student_play():
    """Exhaustive: student (X) tries every move sequence, AI (O) always answers with minimax."""
    wins = {"X": 0, "O": 0}

    def winner(b):
        for a, c, d in [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]:
            if b[a] and b[a] == b[c] == b[d]:
                return b[a]
        return None

    def student_turn(b):
        w = winner(b)
        if w or all(b):
            if w:
                wins[w] += 1
            return
        for p in range(9):
            if b[p] == "":
                nb = list(b)
                nb[p] = "X"
                if winner(nb) or all(nb):
                    if winner(nb):
                        wins["X"] += 1
                    continue
                _, ai = minimax(nb, True)
                nb[ai] = "O"
                student_turn(nb)

    student_turn([""] * 9)
    assert wins["X"] == 0, "the AI lost a game"
