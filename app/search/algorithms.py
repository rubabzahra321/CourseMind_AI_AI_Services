
from collections import deque
from math import inf


def bfs_prerequisites(target_topic_id, prerequisites):
    """
    prerequisite rows:
    {
        "topic_id": "advanced_topic_id",
        "prerequisite_topic_id": "basic_topic_id"
    }

    Returns the prerequisite chain from foundational topic to target.
    """
    graph = {}

    for edge in prerequisites:
        graph.setdefault(edge["topic_id"], []).append(
            edge["prerequisite_topic_id"]
        )

    queue = deque([(target_topic_id, [target_topic_id])])
    visited = {target_topic_id}

    while queue:
        current, path = queue.popleft()

        for prerequisite in graph.get(current, []):
            if prerequisite in visited:
                continue

            new_path = path + [prerequisite]

            # The first prerequisite-free node gives the shortest chain.
            if prerequisite not in graph or not graph[prerequisite]:
                return list(reversed(new_path))

            visited.add(prerequisite)
            queue.append((prerequisite, new_path))

    return [target_topic_id]
def bfs_all_prerequisites(target_topic_id, prerequisites):
    """
    Return all reachable prerequisites before the target topic.

    Each row means:
        topic_id requires prerequisite_topic_id

    BFS first collects all reachable topics. A second queue-based
    topological ordering ensures prerequisites appear before topics
    that depend on them.

    The target topic is included as the final item.
    """
    prerequisite_graph = {}

    for edge in prerequisites:
        prerequisite_graph.setdefault(
            str(edge["topic_id"]), []
        ).append(str(edge["prerequisite_topic_id"]))

    # Step 1: BFS to collect every reachable prerequisite.
    queue = deque([str(target_topic_id)])
    reachable = {str(target_topic_id)}

    while queue:
        current = queue.popleft()

        for prerequisite_id in prerequisite_graph.get(current, []):
            if prerequisite_id not in reachable:
                reachable.add(prerequisite_id)
                queue.append(prerequisite_id)

    # Step 2: Reverse the edges:
    # prerequisite -> topics that depend on it.
    dependents = {topic_id: [] for topic_id in reachable}
    indegree = {topic_id: 0 for topic_id in reachable}

    for topic_id in reachable:
        for prerequisite_id in prerequisite_graph.get(topic_id, []):
            if prerequisite_id not in reachable:
                continue

            dependents[prerequisite_id].append(topic_id)
            indegree[topic_id] += 1

    # Step 3: Queue topics that have no prerequisites.
    ready = deque(
        topic_id
        for topic_id in reachable
        if indegree[topic_id] == 0
    )

    ordered = []

    while ready:
        current = ready.popleft()
        ordered.append(current)

        for dependent_id in dependents[current]:
            indegree[dependent_id] -= 1

            if indegree[dependent_id] == 0:
                ready.append(dependent_id)

    # A cycle means no valid prerequisite-first order exists.
    if len(ordered) != len(reachable):
        raise ValueError(
            "A cycle was found in the prerequisite graph. "
            "Correct the prerequisite relationships before generating a plan."
        )

    return ordered

def dfs_topic_subtree(root_topic_id, topics):
    """
    Collect a topic and all descendants using DFS.
    Each topic must contain id and parent_topic_id.
    """
    children = {}

    for topic in topics:
        children.setdefault(topic.get("parent_topic_id"), []).append(topic)

    result = []
    visited = set()

    def visit(topic_id, depth):
        if topic_id in visited:
            return

        visited.add(topic_id)
        result.append({"topic_id": topic_id, "depth": depth})

        for child in children.get(topic_id, []):
            visit(child["id"], depth + 1)

    visit(root_topic_id, 0)
    return result


def greedy_best_first(topics, limit=None):
    """
    Greedy Best-First Search:
    At each step, choose the remaining topic with the highest heuristic.
    The heuristic is calculated from frequency and weakness.
    """
    remaining = list(topics)
    ordered = []

    while remaining and (limit is None or len(ordered) < limit):
        best = max(
            remaining,
            key=lambda item: float(item.get("utility_score", 0))
        )
        ordered.append(best)
        remaining.remove(best)

    return ordered


def minimax(board, maximizing_player, alpha=-inf, beta=inf):
    """
    Tic-Tac-Toe Minimax with Alpha-Beta Pruning.

    Board values: "", "X", "O"
    X = student
    O = AI

    Returns (score, best_position).
    """
    winning_lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6),
    ]

    def winner():
        for a, b, c in winning_lines:
            if board[a] and board[a] == board[b] == board[c]:
                return board[a]
        return None

    def terminal_score(depth):
        result = winner()

        if result == "O":
            return 10 - depth
        if result == "X":
            return depth - 10
        if all(board):
            return 0
        return None

    score = terminal_score(0)
    if score is not None:
        return score, None

    best_position = None

    if maximizing_player:
        best_score = -inf

        for position in range(9):
            if board[position] != "":
                continue

            board[position] = "O"
            child_score, _ = minimax(
                board, False, alpha, beta
            )
            board[position] = ""

            if child_score > best_score:
                best_score = child_score
                best_position = position

            alpha = max(alpha, best_score)
            if beta <= alpha:
                break

        return best_score, best_position

    best_score = inf

    for position in range(9):
        if board[position] != "":
            continue

        board[position] = "X"
        child_score, _ = minimax(
            board, True, alpha, beta
        )
        board[position] = ""

        if child_score < best_score:
            best_score = child_score
            best_position = position

        beta = min(beta, best_score)
        if beta <= alpha:
            break

    return best_score, best_position
