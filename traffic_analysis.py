import networkx as nx
import numpy as np
from scipy.optimize import minimize
import argparse
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Traffic Equilibrium Analysis')
    parser.add_argument('graph_file', help='GML graph file')
    parser.add_argument('n_vehicles', type=int, help='Number of vehicles')
    parser.add_argument('source', type=int, help='Source node')
    parser.add_argument('target', type=int, help='Target node')
    parser.add_argument('--plot', action='store_true', help='Plot the graph')
    
    args = parser.parse_args()
    
    try:
        # Read graph
        G = nx.read_gml(args.graph_file)
        edges = list(G.edges())
        
        if not G.is_directed():
            print(f"Error: file {args.graph_file} is not directed")
            return
        if G.number_of_edges() == 0:
            print(f"Error: file {args.graph_file} has no edges")
            return
        
        print(f"Graph {args.graph_file} has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

        if str(args.source) not in G:
            print(f"Error: file {args.graph_file} does not contain source node")
            print(f"Available nodes: {list(G.nodes())}")
            return
        
        if str(args.target) not in G:
            print(f"Error: file {args.graph_file} does not contain target node")
            print(f"Available nodes: {list(G.nodes())}")
            return

        # Check that all edges have required parameters 'a' and 'b'
        for u, v in G.edges():
            edge_data = G[u][v]
            if 'a' not in edge_data or 'b' not in edge_data:
                print(f"Error: file {args.graph_file} does not contain edge parameters 'a' and 'b'")
                return False

        
        print("✓ All edges have required parameters 'a' and 'b'")

        # Get cost parameters
        a = []
        b = []
        for u, v in edges:
            edge_data = G[u][v]
            a.append(edge_data.get('a', 0))
            b.append(edge_data.get('b', 0))
        
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        
        # Cost functions
        def edge_cost(flow, idx):
            return a[idx] * flow + b[idx]
        
        def total_cost(flows):
            return sum(flows[i] * edge_cost(flows[i], i) for i in range(len(flows)))
        
        def beckmann_objective(flows):
            return sum(a[i] * flows[i]**2 / 2 + b[i] * flows[i] for i in range(len(flows)))
        
        # Constraints: total flow = n_vehicles, all flows >= 0
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - args.n_vehicles})
        bounds = [(0, args.n_vehicles) for _ in range(len(edges))]
        x0 = np.ones(len(edges)) * args.n_vehicles / len(edges)  # initial guess
        
        # Compute User Equilibrium
        result_ue = minimize(beckmann_objective, x0, method='SLSQP', 
                           bounds=bounds, constraints=constraints)
        ue_flows = result_ue.x if result_ue.success else None
        
        # Compute Social Optimum
        result_so = minimize(total_cost, x0, method='SLSQP', 
                           bounds=bounds, constraints=constraints)
        so_flows = result_so.x if result_so.success else None
        
        # Print results
        print("\n" + "="*50)
        print("TRAFFIC EQUILIBRIUM RESULTS")
        print("="*50)
        print(f"Vehicles: {args.n_vehicles}, Source: {args.source}, Target: {args.target}")
        print(f"\n{'Edge':<8} {'Cost fn':<12} {'UE Flow':<10} {'SO Flow':<10}")
        print("-" * 45)
        
        for i, (u, v) in enumerate(edges):
            ue_val = ue_flows[i] if ue_flows is not None else "FAIL"
            so_val = so_flows[i] if so_flows is not None else "FAIL"
            print(f"({u},{v})   {a[i]:.1f}x+{b[i]:.1f}   {ue_val:<10.3f} {so_val:<10.3f}")
        
        if ue_flows is not None and so_flows is not None:
            ue_cost = total_cost(ue_flows)
            so_cost = total_cost(so_flows)
            print(f"\nUE Total Cost: {ue_cost:.3f}")
            print(f"SO Total Cost: {so_cost:.3f}")
            print(f"Price of Anarchy: {ue_cost/so_cost:.3f}")
        
        # Plot if requested
        if args.plot and ue_flows is not None and so_flows is not None:
            plt.figure(figsize=(12, 5))
            
            # Plot 1: Network
            plt.subplot(1, 2, 1)
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                    node_size=500, font_size=10, arrows=True)
            
            edge_labels = {(u, v): f"{a[i]:.1f}x+{b[i]:.1f}" 
                          for i, (u, v) in enumerate(edges)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
            plt.title("Network with Cost Functions")
            
            # Plot 2: Flow comparison
            plt.subplot(1, 2, 2)
            x = range(len(edges))
            plt.bar([i-0.2 for i in x], ue_flows, 0.4, label='User Equilibrium', alpha=0.7)
            plt.bar([i+0.2 for i in x], so_flows, 0.4, label='Social Optimum', alpha=0.7)
            plt.xticks(x, [f'({u},{v})' for u, v in edges], rotation=45)
            plt.ylabel('Flow')
            plt.title('Flow Comparison')
            plt.legend()
            plt.tight_layout()
            plt.show()
            
    except FileNotFoundError:
        print(f"Error: File {args.graph_file} not found")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()