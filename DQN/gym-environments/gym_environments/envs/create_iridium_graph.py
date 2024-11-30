import networkx as nx

def create_iridium_graph():
    # Initialize graph
    G = nx.Graph()
    
    # Parameters
    num_planes = 6  # Number of orbital planes
    sats_per_plane = 11  # Number of satellites per orbital plane
    total_sats = num_planes * sats_per_plane
    
    # Add nodes for each satellite
    G.add_nodes_from(range(total_sats))
    
    # Create edges
    for plane in range(num_planes):
        for sat in range(sats_per_plane):
            # Current satellite index
            current_sat = plane * sats_per_plane + sat
            
            # Intra-plane links
            next_sat_in_plane = plane * sats_per_plane + (sat + 1) % sats_per_plane
            G.add_edge(current_sat, next_sat_in_plane)
            
            # Inter-plane links (to adjacent planes)
            next_plane = (plane + 1) % num_planes
            prev_plane = (plane - 1) % num_planes  # Wrap-around for the previous plane
            corresponding_next = next_plane * sats_per_plane + sat
            corresponding_prev = prev_plane * sats_per_plane + sat
            G.add_edge(current_sat, corresponding_next)
            G.add_edge(current_sat, corresponding_prev)
    
    return G


if __name__ == "__main__":
    # Generate the Iridium graph
    iridium_graph = create_iridium_graph()

    # Extract and print all nodes
    nodes = list(iridium_graph.nodes)
    print("Nodes:")
    print(nodes)

    # Extract and print all edges
    edges = list(iridium_graph.edges)
    print("\nEdges (source, destination):")
    for edge in edges:
        print(edge)

    # Visualization (optional)
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        nx.draw(iridium_graph, with_labels=True, node_color='lightblue', font_weight='bold', node_size=500)
        plt.title("Iridium Satellite Network Graph")
        plt.show()
    except ImportError:
        print("\nMatplotlib is not installed. Skipping visualization.")
