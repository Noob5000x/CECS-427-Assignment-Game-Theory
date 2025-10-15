# CECS427 Assignment 3: Game Theory
### Names: Daniel Jose Quizon (030462352) & Christella Marie P. Taguicana (031400952)

## Usage Instructions

Ensure that 'networkx', 'numpy', 'scipy', and 'matplotlib' modules are installed.
```
pip install networkx numpy scipy matplotlib
```

The program is executed from the command line with a specific set of required and optional arguments.

python ./traffic_analysis.py <digraph_file.gml> <N_Vehicles> <Source> <Target> [--plot]

- Digraph File (str): the path to the directed graph file in GML format
- N Vehicles (float): the total number of vehicles/flow in the network
- Source (str): the starting node ID for the flow (must match onde ID in GML)
- Target (str): the ending node ID for the flow (must match node ID in GML)
- --plot (flag): if included, generates plots of the network and flow comparison

Examples: 
- python ./traffic_analysis.py traffic.gml 4 0 3 --plot
- python ./traffic_analysis.py traffic2.gml 4 0 3

## Implementation
Parsing and Graph Setup
```
Input Reading: the program uses argparse to read the GML file, the number of vehicles (N), and the source and target node IDs. The IDs are read as strings.

Edge Data: the code iterates through the graph's edges to extract the cost function parameters 'a' and 'b' for the function c(x) = ax + b.
```

Path Flow
```
Path Identification: it uses networkx.all_simple_paths to find every possible simple path from the source to the target

Incidence Matrix (A): it constructs an edge-to-path incidence matrix that links two sets of variables. This helps to define the edge flow vector (x) as a function of the path flow vector.

Optimization variables: the optimizer only works with the path flow variables f, which satisfies flow conservation at all intermediate nodes.
```

Optimization
```
User Equilibrium (UE): the Beckmann Function minimizes the sum of the integrals of the edge cost functions. This finds the Nash Equilibrium, where selfish drivers choose the minimum cost path.

Social Optimum (SO): the Total Cost Function minimizes the sum of flow x cost for every edge. This finds the flow that minimizes the Total System Travel Time.
```

Visualization & Output
```
Result Calculation: The program converts the optimal path flow vectors back into the required edge flow vectors using the incidence matrix.

Summary: The UE Total Cost, SO Total Cost, and Price of Anarchy are calculated.

Plotting: If the --plot flag is used, two charts are generated: the network graph with cost functions and a bar chart comparing the UE and SO edge flows.
```