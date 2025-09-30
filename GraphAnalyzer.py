# run "pip install networkx pandas" in cmd to install packages

import networkx as nx
import pandas as pd
from itertools import combinations, product
from graphviz import Digraph

# Function to compute the neighborhoods
def compute_neighborhoods(graph, vertex):
    first_out = set(graph.successors(vertex))  # First out neighborhood
    second_out = set()  # Second out neighborhood
    first_in = set(graph.predecessors(vertex))  # First in neighborhood

    # Collect second out neighborhood
    for v in first_out:
        #print(f"Vertex: {v}, Successors: {len(set(graph.successors(v)))}")
        second_out.update(graph.successors(v))

    # Remove vertices from the second out neighborhood that are in the first out neighborhood
    second_out.difference_update(first_out)

    return first_out, second_out, first_in

# Functions to check Seymour and Sullivan conditions
def is_seymour(first_out, second_out):
    return len(second_out) >= len(first_out)

def is_sullivan(first_in, second_out):
    return len(second_out) >= len(first_in)

# Main GraphAnalyzer class
def analyze_graph(graph, verbose=False):
    data = []
    sey_count = 0
    sul_count = 0
    
    # Iterate over all vertices in the graph
    for vertex in sorted(graph.nodes()):
        first_out, second_out, first_in = compute_neighborhoods(graph, vertex)
        seymour = is_seymour(first_out, second_out)
        if(seymour):
            sey_count += 1
        sullivan = is_sullivan(first_in, second_out)
        if(sullivan):
            sul_count += 1

        # Store results in a dictionary
        data.append({
            "Vertex": vertex,
            "|N+|": len(first_out),
            "|N++|": len(second_out),
            "|N-|": len(first_in),
            "Is Seymour": seymour,
            "Is Sullivan": sullivan
        })
    
    # Create a pandas DataFrame for easy display
    if verbose:
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        print()
        print(f"Seymour: {sey_count}")
        print(f"Sullivan: {sul_count}")
    return sey_count

def is_simple_digraph(G):
    return not nx.number_of_selfloops(G)

# is d regular checks for a degree regular graph ex. 4-regular (quartic)
def is_d_regular(G, d):
    for v in G.nodes():
        if d != G.degree(v):
            return False
    return True

# Uses Havel–Hakimi Algorithm applied to digraphs
def generate_graph(n, out_degree, d):
    """
    Deterministically construct one simple digraph with:
      - n vertices (1..n)
      - prescribed outdegree sequence
      - total degree d
    """
    if len(out_degree) != n:
        raise ValueError("out_degree distribution must have length n")

    indegree = [d - o for o in out_degree]
    if sum(out_degree) != sum(indegree):
        raise ValueError("Degree sequences not graphical")

    G = nx.DiGraph()
    G.add_nodes_from(range(1, n+1))

    # Work with a list of [vertex, out_remaining, in_remaining]
    degree_info = [[v+1, out_degree[v], indegree[v]] for v in range(n)]

    # While there are vertices with out_remaining > 0
    while any(o > 0 for _, o, _ in degree_info):
        # Pick vertex with largest out_remaining
        degree_info.sort(key=lambda x: -x[1])  
        u, out_u, in_u = degree_info[0]

        if out_u == 0:
            break

        # Find candidates for targets: vertices not u, still need indegree, and no existing edge
        candidates = [v for v, o, i in degree_info if v != u and i > 0 and not G.has_edge(u, v)]

        if len(candidates) < out_u:
            raise ValueError("No valid digraph exists with this degree sequence")

        # Take the first out_u candidates
        for v in candidates[:out_u]:
            G.add_edge(u, v)
            # Update indegree
            for entry in degree_info:
                if entry[0] == v:
                    entry[2] -= 1
                    break

        # Update outdegree
        degree_info[0][1] = 0

    return G

def generate_graphs(n, d, out_degree=None, sink=True, source=True):
    """
    Generate all simple digraphs with:
      - n vertices (1..n)
      - total degree d for each vertex
      - optional outdegree sequence (multiset of length n)
      - optional sink/source restrictions

    Parameters
    ----------
    n : int
        Number of vertices
    d : int
        Degree of each vertex
    out_degree : list[int] or None
        Outdegree distribution (length n). If None, all distributions are allowed.
    sink : bool
        If False, disallow vertices with outdegree 0.
    source : bool
        If False, disallow vertices with outdegree d.
    """

    if out_degree is None:
        # Generate all possible outdegree distributions
        # Each entry in [0..d], sum(out) = n*d/2 if indegree matches
        for out_seq in product(range(d+1), repeat=n):
            if not sink and any(o == 0 for o in out_seq):
                continue
            if not source and any(o == d for o in out_seq):
                continue

            indegree = [d - o for o in out_seq]
            if sum(out_seq) != sum(indegree):
                continue  # must balance

            vertices = list(range(1, n+1))
            degree_info = {v: [out_seq[v-1], indegree[v-1]] for v in vertices}

            G = nx.DiGraph()
            G.add_nodes_from(vertices)

            yield from digraph_backtrack(G, degree_info)

    else:
        if len(out_degree) != n:
            raise ValueError("out_degree distribution must have length n")

        if not sink and any(o == 0 for o in out_degree):
            return
        if not source and any(o == d for o in out_degree):
            return

        indegree = [d - o for o in out_degree]
        if sum(out_degree) != sum(indegree):
            raise ValueError("Degree sequences not graphical")

        vertices = list(range(1, n+1))
        degree_info = {v: [out_degree[v-1], indegree[v-1]] for v in vertices}

        G = nx.DiGraph()
        G.add_nodes_from(vertices)

        yield from digraph_backtrack(G, degree_info)


def digraph_backtrack(G, degree_info):
    # If all degrees are satisfied, yield solution
    if all(out == 0 and inn == 0 for out, inn in degree_info.values()):
        yield G.copy()
        return

    # Pick a vertex with remaining outdegree
    u = next((v for v, (out, _) in degree_info.items() if out > 0), None)
    if u is None:
        return  # dead end

    out_u, _ = degree_info[u]

    # Possible targets for u
    candidates = [
        v for v, (out, inn) in degree_info.items()
        if v != u and inn > 0 and not G.has_edge(u, v) and not G.has_edge(v, u)
    ]

    if len(candidates) < out_u:
        return  # can't satisfy outdegree

    for choice in combinations(candidates, out_u):
        # Apply edges
        for v in choice:
            G.add_edge(u, v)
            degree_info[u][0] -= 1
            degree_info[v][1] -= 1

        # Recurse
        yield from digraph_backtrack(G, degree_info)

        # Backtrack
        for v in choice:
            G.remove_edge(u, v)
            degree_info[u][0] += 1
            degree_info[v][1] += 1

def visualize_graph(G):
    # Use circular layout
    dot = Digraph(engine="circo")

    # Increase size and resolution
    dot.attr(size="8,8", dpi="300")

    # Add edges
    for u, v in G.edges():
        dot.edge(str(u), str(v))

    dot.render("circle_graph", format="png", view=True)


# Example usage
if __name__ == "__main__":
    # Create a simple directed graph
    #G = nx.DiGraph()
    # edit this line with edpythges from your specfic graph
    # Perfect n=4
    #G.add_edges_from([(1, 2), (1, 5), (2,3), (3,4), (3,5), (4,1), (5,4), (5,2)])
    # Perfect n=5
    #G.add_edges_from([(1, 2), (1, 5), (2,3), (2,6), (3,4), (4,5), (4,6), (5,6), (6,1), (6,3)])
    # Lowest found for n=4
    #G.add_edges_from([(1, 2), (1, 5), (2,3), (3,4), (5,3), (4,1), (4,5), (5,2)])

    #generate quartic graph
    """ G = generate_graph(7, [1, 1, 2, 2, 2, 3, 3], 4)

    analyze_graph(G) """
    n = 8
    out_distribution = [1,1,2,2,2,3,3]
    d = 4
    total = 0

    for i, G in enumerate(generate_graphs(n, d, sink=False, source=False), 1):
        total = total + 1
        if i == 1:
            lowSey = analyze_graph(G)
            lowSeyG = G
        else:
            sey_of_G = analyze_graph(G)
            if sey_of_G < lowSey:
                print("NEW LOW UNLOCKED")
                lowSey = sey_of_G
                lowSeyG = G
                print(f"Graph {i}: {list(G.edges())}")
    
    print()
    print()
    print(f"Final low: {lowSey}")
    print(f"Edge Set: {list(lowSeyG.edges())}")
    print(f"Graphs Checked: {i}")
    #visualize_graph(lowSeyG)
    analyze_graph(lowSeyG, verbose=True)

