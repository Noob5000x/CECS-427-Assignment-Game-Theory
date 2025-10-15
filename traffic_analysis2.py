import networkx as nx
import numpy as np
from scipy.optimize import minimize
import argparse
import matplotlib.pyplot as plt

def total_cost_path_flow(path_flows, edge_to_path_matrix, a, b):
    edge_flows = edge_to_path_matrix @ path_flows
    cost = np.sum(edge_flows * (a * edge_flows + b))
    return cost

def beckmann_objective_path_flow(path_flows, edge_to_path_matrix, a, b):
    edge_flows = edge_to_path_matrix @ path_flows
    beckmann = np.sum(a * edge_flows**2 / 2 + b * edge_flows)
    return beckmann

def main():
    parser = argparse.ArgumentParser(description='Traffic Equilibrium Analysis')
    parser.add_argument('graph_file', help='GML graph file')
    parser.add_argument('n_vehicles', type=float, help='Number of vehicles')
    parser.add_argument('source', type=str, help='Source node')
    parser.add_argument('target', type=str, help='Target node')
    parser.add_argument('--plot', action='store_true', help='Plot the graph')
    
    args = parser.parse_args()
    
    try:
        # Read graph
        G = nx.read_gml(args.graph_file)

        source = args.source
        target = args.target
        
        if source not in G or target not in G:
            print(f"Error: Source ({source}) or Target ({target}) not in graph.")
            print(f"Available nodes: {list(G.nodes())}")
            return
               
        print(f"Graph {args.graph_file} has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

        # Check that all edges have required parameters 'a' and 'b'
        for u, v in G.edges():
            edge_data = G[u][v]
            if 'a' not in edge_data or 'b' not in edge_data:
                print(f"Error: file {args.graph_file} does not contain edge parameters 'a' and 'b'")
                return
            try:
                edge_data['a'] = float(edge_data['a'])
                edge_data['b'] = float(edge_data['b'])
            except ValueError:
                print(f"Error: Edge parameters 'a' or 'b' on edge ({u}, {v}) are not valid numbers.")
                return

        
        print("✓ All edges have required parameters 'a' and 'b'")

        all_paths = list(nx.all_simple_paths(G, source=source, target=target))
        if not all_paths:
            print(f"Error: No path found from source {source} to target {target}.")
            return

        n_edges = G.number_of_edges()
        edges = list(G.edges())
        n_paths = len(all_paths)

        # Get cost parameters
        a = np.array([G[u][v]['a'] for u, v, in edges], dtype=float)
        b = np.array([G[u][v]['b'] for u, v, in edges], dtype=float)

        edge_to_path_matrix = np.zeros((n_edges, n_paths))
        edge_to_index = {edge: i for i, edge in enumerate(edges)}

        for j, path in enumerate(all_paths):
            path_edges = list(zip(path[:-1], path[1:]))
            for u, v in path_edges:
                edge_index = edge_to_index[(u, v)]
                edge_to_path_matrix[edge_index, j] = 1
        
        # Constraints: total flow = n_vehicles, all flows >= 0
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - args.n_vehicles})
        bounds = [(0, args.n_vehicles) for _ in range(n_paths)]
        x0 = np.ones(n_paths) * args.n_vehicles / n_paths  # initial guess
        
        # Compute User Equilibrium
        result_ue = minimize(beckmann_objective_path_flow, x0, args= (edge_to_path_matrix, a, b), method='SLSQP', 
                           bounds=bounds, constraints=constraints)
        ue_path_flows = result_ue.x if result_ue.success else None
        ue_edge_flows = edge_to_path_matrix @ ue_path_flows if ue_path_flows is not None else None
        
        # Compute Social Optimum
        result_so = minimize(total_cost_path_flow, x0, args=(edge_to_path_matrix, a, b), method='SLSQP', 
                           bounds=bounds, constraints=constraints)
        so_path_flows = result_so.x if result_so.success else None
        so_edge_flows = edge_to_path_matrix @ so_path_flows if so_path_flows is not None else None
        
        # Print results
        print("\n" + "="*70)
        print("TRAFFIC EQUILIBRIUM RESULTS")
        print("="*70)
        print(f"Vehicles: {args.n_vehicles:.2f}, Source: {source}, Target: {target}")
        print(f"\n{'Edge':<8} {'Cost fn':<12} {'UE Flow':<10} {'SO Flow':<10}")
        print("-" * 45)
        
        for i, (u, v) in enumerate(edges):
            cost_fn = f"{a[i]:.1f}x+{b[i]:.1f}"
            ue_val = ue_edge_flows[i] if ue_edge_flows is not None else "FAIL"
            so_val = so_edge_flows[i] if so_edge_flows is not None else "FAIL"

            if isinstance(ue_val, str):
                print(f"({u}, {v})  {cost_fn:<12} {ue_val:<10} {so_val:<10}")
            else:
                print(f"({u},{v})   {cost_fn:<12} {ue_val:<10.3f} {so_val:<10.3f}")
        
        print("\n" + "-"*45)
        print(f"{'Path':<12} {'UE Path Flow':<16} {'SO Path Flow':<16}")
        print("-" * 45)

        for j, path in enumerate(all_paths):
            path_str = "->".join(path)
            ue_path_val = ue_path_flows[j] if ue_path_flows is not None else "FAIL"
            so_path_val = so_path_flows[j] if so_path_flows is not None else "FAIL"

            if isinstance(ue_path_val, str):
                print(f"{path_str:<12} {ue_path_val:<16} {so_path_val:<16}")
            else:
                print(f"{path_str:<12} {ue_path_val:<16.3f} {so_path_val:<16.3f}")

        if ue_edge_flows is not None and so_edge_flows is not None:
            ue_cost = total_cost_path_flow(ue_path_flows, edge_to_path_matrix, a, b)
            so_cost = total_cost_path_flow(so_path_flows, edge_to_path_matrix, a, b)
            
            print("\n" + "="*45)
            print("SUMMARY")
            print("="*45)
            print(f"\nUE Total Cost: {ue_cost:.3f}")
            print(f"SO Total Cost: {so_cost:.3f}")
            print(f"Price of Anarchy: {ue_cost/so_cost:.3f}")
        
        # Plot if requested
        if args.plot and ue_edge_flows is not None and so_edge_flows is not None:
            plt.figure(figsize=(12, 5))
            
            # Plot 1: Network
            plt.subplot(1, 2, 1)
            try:
                pos = nx.planar_layout(G)
            except nx.NetworkXException:
                pos = nx.spring_layout(G)

            nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                    node_size=800, font_size=10, arrows=True, arrowsize=20)
            
            edge_labels = {(u, v): f"{a[i]:.1f}x+{b[i]:.1f}" 
                          for i, (u, v) in enumerate(edges)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
            plt.title("Network with Cost Functions")
            
            # Plot 2: Flow comparison
            plt.subplot(1, 2, 2)
            edge_names = [f'({u},{v})' for u, v in edges]
            x = np.arange(n_edges)
            width = 0.4

            rects1 = plt.bar(x - width/2, ue_edge_flows, width, label='User Equilibrium', alpha=0.7)
            rects2 = plt.bar(x + width/2, so_edge_flows, width, label='Social Optimum', alpha=0.7)

            plt.ylabel('Vehicle Flow')
            plt.title(f'Flow Comparison (Total Vehicles = {args.n_vehicles:.2f})')
            plt.xticks(x, edge_names, rotation=45, ha="right")
            plt.legend()
            plt.grid(axis='y', linestyle='--')
            plt.tight_layout()
            plt.show()
            
    except FileNotFoundError:
        print(f"Error: File {args.graph_file} not found")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()