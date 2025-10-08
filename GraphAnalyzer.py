# run "pip install networkx pandas" in cmd to install packages

import networkx as nx
import pandas as pd
from itertools import combinations, product
from graphviz import Digraph
import math
import os
import shutil

# Function to compute the neighborhoods
def compute_neighborhoods(graph, vertex):
    first_out = set(graph.successors(vertex))  # First out neighborhood
    second_out = set()  # Second out neighborhood
    first_in = set(graph.predecessors(vertex))  # First in neighborhood

    # Collect second out neighborhood
    for v in first_out:
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
    return sey_count, sul_count

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
    
    # Deterministically construct one simple digraph with:
    #   - n vertices (1..n)
    #   - prescribed outdegree sequence
    #   - total degree d
    
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
    
    # Generate all simple digraphs with:
    #   - n vertices (1..n)
    #   - total degree d for each vertex
    #   - optional outdegree sequence (multiset of length n)
    #   - optional sink/source restrictions

    # Parameters
    # ----------
    # n : int
    #     Number of vertices
    # d : int
    #     Degree of each vertex
    # out_degree : list[int] or None
    #     Outdegree distribution (length n). If None, all distributions are allowed.
    # sink : bool
    #     If False, disallow vertices with outdegree 0.
    # source : bool
    #     If False, disallow vertices with outdegree d.
    

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

def generate_antiprism_graph(n):
    # Construct the undirected n-antiprism graph.
    G = nx.Graph()
    # top cycle: 0..n-1
    top = list(range(1, n+1))
    bottom = list(range(n+1, 2*n+1))
    
    # add cycle edges
    for cycle in [top, bottom]:
        for i in range(n):
            G.add_edge(cycle[i], cycle[(i+1) % n])
    
    # add cross edges
    for i in range(n):
        G.add_edge(top[i], bottom[i])
        G.add_edge(top[i], bottom[(i+1) % n])
    
    return G

def generate_antiprism_orientations(n, sink=True, source=True):
    
    # Generate all orientations of the n-antiprism graph.
    
    # Parameters
    # ----------
    # n : int
    #     Base cycle size (graph will have 2n vertices).
    # sink : bool
    #     If False, disallow any orientation with a vertex of outdegree 0.
    # source : bool
    #     If False, disallow any orientation with a vertex of outdegree 4 (quartic).
    
    UG = generate_antiprism_graph(n)
    edges = list(UG.edges())
    d = 4  # antiprism graphs are quartic
    
    for choice in product([0, 1], repeat=len(edges)):
        DG = nx.DiGraph()
        DG.add_nodes_from(UG.nodes())
        
        # orient edges according to choice
        for (u, v), orient in zip(edges, choice):
            if orient == 0:
                DG.add_edge(u, v)
            else:
                DG.add_edge(v, u)
        
        # apply sink/source restrictions
        if not sink and any(DG.out_degree(v) == 0 for v in DG.nodes()):
            continue
        if not source and any(DG.out_degree(v) == d for v in DG.nodes()):
            continue
        
        yield DG

def visualize_graph(G, title, dest=".", verbose=False):
    # Use circular layout
    dot = Digraph(engine="circo")

    # Increase size and resolution
    dot.attr(size="8,8", dpi="300")

    # Add edges
    for u, v in G.edges():
        dot.edge(str(u), str(v))

    dot.render(title, directory=dest, format="png", view=verbose)

def rotation_cost(offset, n, outer_positions, G):
    
    # Compute alignment cost of placing inner cycle at a given rotation offset.
    # The cost measures squared distance between adjacent outer/inner nodes.

    cost = 0
    for i in range(n):
        angle = 2 * math.pi * i / n + offset
        xi, yi = math.cos(angle), math.sin(angle)
        inner_node = n + i + 1
        for u, v in G.edges():
            if u == inner_node and v in outer_positions:
                xo, yo = outer_positions[v]
                cost += (xi - xo)**2 + (yi - yo)**2
            if v == inner_node and u in outer_positions:
                xo, yo = outer_positions[u]
                cost += (xi - xo)**2 + (yi - yo)**2
    return cost

def visualize_antiprism(G, n, title, dest=".", verbose=False):

    # Visualize antiprism-like digraph with color-coded vertices:
    #   - light blue: Seymour only
    #   - light green: Sullivan only
    #   - gold: both Seymour & Sullivan
    #   - gray: neither

    dot = Digraph(engine="neato")
    dot.attr(size="8,8", dpi="300", splines="line", overlap="false")

    # Determine color for each vertex
    color_map = {}
    for v in G.nodes():
        first_out, second_out, first_in = compute_neighborhoods(G, v)
        seymour = is_seymour(first_out, second_out)
        sullivan = is_sullivan(first_in, second_out)

        if seymour and sullivan:
            color_map[v] = "#F4D03F"         # both
        elif seymour:
            color_map[v] = "lightblue"    # seymour only
        elif sullivan:
            color_map[v] = "lightgreen"   # sullivan only
        else:
            color_map[v] = "gray"       # neither

    # Outer cycle (1..n)
    R_outer = 2.0
    outer_positions = {}
    for i in range(n):
        angle = 2 * math.pi * i / n + math.pi / 4
        x = R_outer * math.cos(angle)
        y = R_outer * math.sin(angle)
        node = i + 1
        outer_positions[node] = (x, y)
        dot.node(str(node), pos=f"{x},{y}!", shape="circle", style="filled", fillcolor=color_map[node])

    # Find best rotation for inner cycle
    best_offset, best_cost = None, float("inf")
    for k in range(n):
        offset = 2 * math.pi * k / n
        cost = rotation_cost(offset, n, outer_positions, G)
        if cost < best_cost:
            best_cost, best_offset = cost, offset

    # Inner cycle (n+1..2n)
    R_inner = 1.0
    for i in range(n):
        angle = 2 * math.pi * i / n + best_offset
        x = R_inner * math.cos(angle)
        y = R_inner * math.sin(angle)
        node = n + i + 1
        dot.node(str(node), pos=f"{x},{y}!", shape="circle", style="filled", fillcolor=color_map[node])

    # Add edges
    for u, v in G.edges():
        dot.edge(str(u), str(v), arrowsize="0.7")

    dot.render(title, directory=dest, format="png", view=verbose)



if __name__ == "__main__":
    # n = number of vertices (for use in generate_graphs)
    n = None
    # For generating antiprism orientations, use n_antiprism
    n_antiprism = 8
    out_dist = [1,3,2,1,3,2]
    d = 4
    total = 0
    low_sey_counter = 0
    low_sey_given_sul = n_antiprism*2 + 1
    low_sul_given_sey = n_antiprism*2 + 1
    lowSey = n_antiprism*2 + 1
    lowSul = n_antiprism*2 + 1
    base_dir = os.getcwd()
    sey_dir = os.path.join(base_dir, f"Minimal Seymour {n_antiprism}-antiprism")
    sey_sul_dir = os.path.join(sey_dir, "Minimal Sullivan")
    sul_dir = os.path.join(base_dir, f"Minimal Sullivan {n_antiprism}-antiprism")
    sul_sey_dir = os.path.join(sul_dir, "Minimal Seymour")
    
    if n_antiprism is not None:
        for i, G in enumerate(generate_antiprism_orientations(n_antiprism, source=False, sink=False), 1):
            sey, sul = analyze_graph(G, verbose=False)

            # --- CASE 1: New minimal Seymour ---
            if sey < lowSey:
                lowSey = sey
                low_sul_given_sey = sul

                # Reset minimal Seymour directory
                shutil.rmtree(sey_dir, ignore_errors=True)
                os.makedirs(sey_sul_dir, exist_ok=True)

                visualize_antiprism(G, n_antiprism, title=f"Graph {i}", dest=sey_sul_dir, verbose=False)

            # --- CASE 2: Same minimal Seymour, smaller Sullivan given minimal Seymour ---
            elif sey == lowSey and sul < low_sul_given_sey:
                low_sul_given_sey = sul

                # Move old files out of subfolder to main folder
                for f in os.listdir(sey_sul_dir):
                    shutil.move(os.path.join(sey_sul_dir, f), sey_dir)

                # Clear subfolder and save this new one
                shutil.rmtree(sey_sul_dir, ignore_errors=True)
                os.makedirs(sey_sul_dir, exist_ok=True)

                visualize_antiprism(G, n_antiprism, title=f"Graph {i}", dest=sey_sul_dir, verbose=False)
            
            # --- CASE 3: Same minimal Seymour, same Sullivan given minimal Seymour ---
            elif sey == lowSey and sul == low_sul_given_sey:
                visualize_antiprism(G, n_antiprism, title=f"Graph {i}", dest=sey_sul_dir, verbose=False)

            # --- CASE 4: Same minimal Seymour, larger Sullivan given minimal Seymour ---
            elif sey == lowSey:
                visualize_antiprism(G, n_antiprism, title=f"Graph {i}", dest=sey_dir, verbose=False)

            # --- CASE 5: New minimal Sullivan ---
            if sul < lowSul:
                lowSul = sul
                low_sey_given_sul = sey

                # Reset minimal Sullivan directory
                shutil.rmtree(sul_dir, ignore_errors=True)
                os.makedirs(sul_sey_dir, exist_ok=True)

                visualize_antiprism(G, n_antiprism, title=f"Graph {i}", dest=sul_sey_dir, verbose=False)

            # --- CASE 6: Same minimal Sullivan, smaller Seymour given minimal Sullivan ---
            elif sul == lowSul and sey < low_sey_given_sul:
                low_sey_given_sul = sey

                # Move old files out of subfolder to main folder
                for f in os.listdir(sul_sey_dir):
                    shutil.move(os.path.join(sul_sey_dir, f), sul_dir)

                # Clear subfolder and save this new one
                shutil.rmtree(sul_sey_dir, ignore_errors=True)
                os.makedirs(sul_sey_dir, exist_ok=True)

                visualize_antiprism(G, n_antiprism, title=f"Graph {i}", dest=sul_sey_dir, verbose=False)

            # --- CASE 7: Same minimal Sullivan, same Seymour given minimal Sullivan ---
            elif sul == lowSul and sey == low_sey_given_sul:
                visualize_antiprism(G, n_antiprism, title=f"Graph {i}", dest=sul_sey_dir, verbose=False)

            # --- CASE 8: Same minimal Sullivan, larger Seymour given minimal Sullivan ---
            elif sul == lowSul:
                visualize_antiprism(G, n_antiprism, title=f"Graph {i}", dest=sul_dir, verbose=False)

    if n is not None:
        lowSey = n + 1
        for i, G in enumerate(generate_graphs(n, d, sink=False, source=False), 1):
            total = total + 1
            sey_of_G = analyze_graph(G)
            if sey_of_G < lowSey:
                print("NEW LOW UNLOCKED")
                lowSey = sey_of_G
                lowSeyG = G
                print(f"Graph {i}: {list(G.edges())}")
            # if i % 1000 == 0:
            #     print(f"Graphs Checked: {i}, Current Low: {lowSey}")


    
    print()
    print()
    print(f"Graphs Checked: {i}")
    print(f"Final Seymour low: {lowSey}")
    print()
    print(f"Lowest Sullivan given Seymour {low_sul_given_sey}")
    # print(f"Edge Set: {list(lowSeyG.edges())}")
    # visualize_antiprism(lowSeyG, n_antiprism, f"A minimal Seymour Graph on a {n_antiprism}-antiprism", verbose=True)
    print(f"Final Sullivan low: {lowSul}")
    print()
    print(f"Lowest Seymour given Sullivan {low_sey_given_sul}")

