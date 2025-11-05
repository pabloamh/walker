# wanderer/utils.py
from typing import List, Tuple, Any, Dict

def group_pairs(pairs: List[Tuple[int, int]], items: List[Any]) -> List[List[Any]]:
    """
    Groups items based on a list of index pairs using a Disjoint Set Union (DSU) data structure.
    This is useful for finding connected components in a graph of similarities.

    Args:
        pairs: A list of tuples, where each tuple (i, j) represents a connection
               between the item at index i and the item at index j.
        items: The list of items to be grouped.

    Returns:
        A list of lists, where each inner list is a group of connected items.
    """
    parent = list(range(len(items)))
    def find(i: int) -> int:
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int):
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    for i, j in pairs:
        union(i, j)

    groups: Dict[int, List[Any]] = {}
    for i in range(len(items)):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(items[i])
    
    return [group for group in groups.values() if len(group) > 1]