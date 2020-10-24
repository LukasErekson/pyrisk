from world import CONNECT, T_INDEX
import networkx as nx

# Initialize a graph of the risk map
GRAPH = nx.Graph()

# Get a list of territories
TERR = list(T_INDEX.keys())

# Get a list of edges from the CONNECT string
EDGES = []
for line in CONNECT.split('\n'):
    if line != '':
        l = line.split('--')
        EDGES += [(l[i], l[i+1]) for i in range(len(l)-1)]

# Populate the graph
GRAPH.add_nodes_from(TERR)
GRAPH.add_edges_from(EDGES)

debug=False

def get_graph_features(row):
    """
    Returns graph-related features given the current state of the game:

    Parameters:
        row (list, len=num_players*42): a list of how many troops each player has in each territory
                                        can be obtained from the dataframe easily

    Returns:
        player_cut_edges: number of boundary edges which cross into the player's controlled area
        player_number_boundary_nodes: number of territories that make up the boundary of the player's controlled area
        player_boundary_fortifications: total number of troops on the boundary of the player's controlled area
        player_average_boundary_fortifications: average number of troops in each boundary territory
        player_connected_components: number of connected components in the area the player controls

    Each item returned is a list of length num_players, which has each of the features calculated for each player
    """

    # Make a copy of the risk graph
    g = GRAPH.copy()
    n = len(row)//42

    # Add troops and player info to this graph
    for i in range(42):
        r = row[i*n:(i+1)*n]
        g.nodes[TERR[i]]['troops'] = max(r)
        g.nodes[TERR[i]]['player'] = r.index(g.nodes[TERR[i]]['troops'])

    # Initialize feature containers
    player_cut_edges = [0]*n
    player_boundary_nodes = [set() for i in range(n)] # helper feature
    player_number_boundary_nodes = [0]*n
    player_boundary_fortifications = [0]*n
    player_average_boundary_fortifications = [0]*n
    player_connected_components = [0]*n

    # Iterate through edges
    for edge in g.edges:
        p1, p2 = g.nodes[edge[0]]['player'],  g.nodes[edge[1]]['player']
        if p1 != p2:
            # Update cut edge counts
            player_cut_edges[p1] += 1
            player_cut_edges[p2] += 1

            # Update boundary nodes
            player_boundary_nodes[p1].add(edge[0])
            player_boundary_nodes[p2].add(edge[1])

            if debug: 
                print(player_boundary_nodes)

    total_cut_edges = sum(player_cut_edges)//2

    # Iterate through players
    for i in range(n):

        # Get the subgraph for each player
        player_graph = g.subgraph([j for j in g.nodes if g.nodes[j]['player'] == i])
        player_connected_components[i] = len(list(nx.connected_components(player_graph)))

        # Update boundary node counts
        player_number_boundary_nodes[i] = len(player_boundary_nodes[i])

        # Iterate through boundary nodes
        for n in player_boundary_nodes[i]:
            player_boundary_fortifications[i] += g.nodes[n]['troops']

        if player_number_boundary_nodes[i] > 0:
            player_average_boundary_fortifications[i] = player_boundary_fortifications[i]/player_number_boundary_nodes[i]
        else:
            player_average_boundary_fortifications[i] = -1

    return player_cut_edges, player_number_boundary_nodes, player_boundary_fortifications, player_average_boundary_fortifications, player_connected_components
