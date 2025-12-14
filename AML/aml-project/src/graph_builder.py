"""
Network Graph Builder Module
============================
Creates transaction network visualizations using NetworkX:

1. Builds transaction flow graph (accounts as nodes, transactions as edges)
2. Identifies suspicious clusters and communities
3. Highlights money flow paths
4. Generates publication-ready visualizations
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class NetworkMetrics:
    """Network analysis metrics for an account."""
    account_id: str
    in_degree: int
    out_degree: int
    total_degree: int
    in_amount: float
    out_amount: float
    pagerank: float
    betweenness: float
    clustering_coefficient: float
    community_id: int


class TransactionGraphBuilder:
    """
    Builds and visualizes transaction networks.
    
    Creates interactive network graphs showing money flows
    between accounts, with visual encoding of risk levels.
    """
    
    # Color schemes
    RISK_COLORS = {
        'critical': '#E53935',   # Red
        'high': '#FF7043',       # Orange
        'medium': '#FFC107',     # Amber
        'low': '#4CAF50',        # Green
        'unknown': '#9E9E9E'     # Gray
    }
    
    EDGE_COLORS = {
        'normal': '#B0BEC5',
        'suspicious': '#FF5722',
        'structuring': '#E91E63',
        'mule': '#9C27B0',
        'layering': '#3F51B5'
    }
    
    def __init__(self):
        self.graph: nx.DiGraph = None
        self.node_metrics: Dict[str, NetworkMetrics] = {}
        self.communities: Dict[int, Set[str]] = {}
        self.suspicious_paths: List[List[str]] = []
    
    def build_graph(
        self, 
        transactions: pd.DataFrame,
        risk_scores: Optional[pd.DataFrame] = None
    ) -> nx.DiGraph:
        """
        Build a directed graph from transactions.
        
        Nodes = accounts
        Edges = transactions (aggregated)
        Edge weight = total transaction amount
        """
        print("🔍 Building Transaction Network Graph...")
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Aggregate transactions between account pairs
        edge_data = transactions.groupby(['sender_id', 'receiver_id']).agg({
            'amount': ['sum', 'count', 'mean'],
            'tx_id': lambda x: list(x),
            'pattern_type': lambda x: x.mode().iloc[0] if len(x) > 0 else 'normal'
        }).reset_index()
        
        edge_data.columns = ['sender', 'receiver', 'total_amount', 'tx_count', 
                            'avg_amount', 'tx_ids', 'pattern_type']
        
        # Add edges
        for _, row in edge_data.iterrows():
            G.add_edge(
                row['sender'],
                row['receiver'],
                weight=row['total_amount'],
                tx_count=row['tx_count'],
                avg_amount=row['avg_amount'],
                tx_ids=row['tx_ids'],
                pattern_type=row['pattern_type']
            )
        
        # Add node attributes from risk scores
        if risk_scores is not None and not risk_scores.empty:
            risk_dict = risk_scores.set_index('account_id').to_dict('index')
            
            for node in G.nodes():
                if node in risk_dict:
                    G.nodes[node]['risk_score'] = risk_dict[node].get('overall_score', 0)
                    G.nodes[node]['risk_category'] = risk_dict[node].get('risk_category', 'unknown')
                    G.nodes[node]['alert_count'] = risk_dict[node].get('alert_count', 0)
                else:
                    G.nodes[node]['risk_score'] = 0
                    G.nodes[node]['risk_category'] = 'unknown'
                    G.nodes[node]['alert_count'] = 0
        
        self.graph = G
        
        print(f"   ├── Nodes (accounts): {G.number_of_nodes()}")
        print(f"   ├── Edges (transaction flows): {G.number_of_edges()}")
        print(f"   └── Total transaction volume: ${transactions['amount'].sum():,.2f}")
        
        return G
    
    def calculate_network_metrics(self) -> pd.DataFrame:
        """
        Calculate network centrality metrics for all nodes.
        
        Metrics help identify key accounts in the network:
        - PageRank: importance based on connections
        - Betweenness: accounts that bridge other accounts
        - Clustering: how connected neighbors are
        """
        print("\n🔍 Calculating Network Metrics...")
        
        G = self.graph
        
        # Calculate centrality metrics
        pagerank = nx.pagerank(G, weight='weight')
        betweenness = nx.betweenness_centrality(G, weight='weight')
        
        # Clustering coefficient (for undirected version)
        G_undirected = G.to_undirected()
        clustering = nx.clustering(G_undirected)
        
        # Detect communities using Louvain method
        try:
            from networkx.algorithms import community
            communities = community.louvain_communities(G_undirected, seed=42)
            
            # Map node to community
            node_community = {}
            for i, comm in enumerate(communities):
                self.communities[i] = comm
                for node in comm:
                    node_community[node] = i
        except:
            node_community = {node: 0 for node in G.nodes()}
            self.communities[0] = set(G.nodes())
        
        # Build metrics for each node
        metrics_list = []
        
        for node in G.nodes():
            in_edges = G.in_edges(node, data=True)
            out_edges = G.out_edges(node, data=True)
            
            in_amount = sum(d.get('weight', 0) for _, _, d in in_edges)
            out_amount = sum(d.get('weight', 0) for _, _, d in out_edges)
            
            metrics = NetworkMetrics(
                account_id=node,
                in_degree=G.in_degree(node),
                out_degree=G.out_degree(node),
                total_degree=G.in_degree(node) + G.out_degree(node),
                in_amount=in_amount,
                out_amount=out_amount,
                pagerank=pagerank.get(node, 0),
                betweenness=betweenness.get(node, 0),
                clustering_coefficient=clustering.get(node, 0),
                community_id=node_community.get(node, 0)
            )
            
            self.node_metrics[node] = metrics
            metrics_list.append(metrics)
        
        # Create DataFrame
        metrics_df = pd.DataFrame([{
            'account_id': m.account_id,
            'in_degree': m.in_degree,
            'out_degree': m.out_degree,
            'total_degree': m.total_degree,
            'in_amount': m.in_amount,
            'out_amount': m.out_amount,
            'pagerank': m.pagerank,
            'betweenness': m.betweenness,
            'clustering_coefficient': m.clustering_coefficient,
            'community_id': m.community_id
        } for m in metrics_list])
        
        print(f"   ├── Communities detected: {len(self.communities)}")
        print(f"   └── High centrality accounts: {len(metrics_df[metrics_df['pagerank'] > metrics_df['pagerank'].quantile(0.95)])}")
        
        return metrics_df
    
    def find_suspicious_paths(
        self, 
        max_path_length: int = 5,
        min_amount: float = 10000
    ) -> List[List[str]]:
        """
        Find suspicious money flow paths.
        
        Looks for:
        - Circular flows (money returning to origin)
        - Long layering chains
        - High-value paths through low-risk accounts
        """
        print("\n🔍 Finding Suspicious Paths...")
        
        G = self.graph
        paths = []
        
        # Find cycles (circular flows)
        try:
            cycles = list(nx.simple_cycles(G))
            for cycle in cycles[:50]:  # Limit to first 50
                if len(cycle) >= 3:
                    # Check total amount in cycle
                    cycle_amount = 0
                    for i in range(len(cycle)):
                        edge_data = G.get_edge_data(cycle[i], cycle[(i+1) % len(cycle)])
                        if edge_data:
                            cycle_amount += edge_data.get('weight', 0)
                    
                    if cycle_amount >= min_amount:
                        paths.append(cycle + [cycle[0]])  # Close the loop for visualization
        except nx.NetworkXError:
            pass
        
        # Find long paths from high-risk to low-risk accounts
        high_risk_nodes = [n for n in G.nodes() 
                          if G.nodes[n].get('risk_category') in ['critical', 'high']]
        
        for source in high_risk_nodes[:20]:  # Limit for performance
            for target in G.nodes():
                if source != target:
                    try:
                        for path in nx.all_simple_paths(G, source, target, cutoff=max_path_length):
                            if len(path) >= 3:
                                # Calculate path amount
                                path_amount = min(
                                    G.get_edge_data(path[i], path[i+1]).get('weight', 0)
                                    for i in range(len(path)-1)
                                )
                                if path_amount >= min_amount:
                                    paths.append(path)
                                    if len(paths) >= 100:  # Limit total paths
                                        break
                    except nx.NetworkXNoPath:
                        pass
                
                if len(paths) >= 100:
                    break
            if len(paths) >= 100:
                break
        
        self.suspicious_paths = paths
        print(f"   └── Suspicious paths found: {len(paths)}")
        
        return paths
    
    def visualize_network(
        self,
        output_path: str = 'network_graph.png',
        figsize: Tuple[int, int] = (20, 16),
        show_labels: bool = True,
        highlight_suspicious: bool = True,
        max_nodes: int = 200,
        title: str = "AML Transaction Network Analysis"
    ) -> plt.Figure:
        """
        Create a publication-ready network visualization.
        
        Features:
        - Node size based on transaction volume
        - Node color based on risk score
        - Edge thickness based on amount
        - Highlighted suspicious clusters
        """
        print("\n🎨 Creating Network Visualization...")
        
        G = self.graph
        
        # Subsample if too many nodes
        if G.number_of_nodes() > max_nodes:
            # Keep highest risk nodes
            nodes_by_risk = sorted(
                G.nodes(data=True),
                key=lambda x: x[1].get('risk_score', 0),
                reverse=True
            )[:max_nodes]
            nodes_to_keep = [n[0] for n in nodes_by_risk]
            G = G.subgraph(nodes_to_keep).copy()
        
        # Set up figure with dark theme for impact
        fig, ax = plt.subplots(figsize=figsize, facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        
        # Calculate layout
        # Use spring layout with weight consideration
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42, weight='weight')
        
        # Prepare node attributes
        node_colors = []
        node_sizes = []
        node_labels = {}
        
        for node in G.nodes():
            risk_cat = G.nodes[node].get('risk_category', 'unknown')
            node_colors.append(self.RISK_COLORS.get(risk_cat, self.RISK_COLORS['unknown']))
            
            # Size based on transaction volume
            metrics = self.node_metrics.get(node)
            if metrics:
                volume = metrics.in_amount + metrics.out_amount
                size = 100 + np.log1p(volume) * 30
            else:
                size = 100
            node_sizes.append(min(size, 1000))  # Cap maximum size
            
            # Labels for high-risk nodes
            if risk_cat in ['critical', 'high']:
                node_labels[node] = node[:10]  # Truncate long IDs
        
        # Prepare edge attributes
        edge_colors = []
        edge_widths = []
        
        for u, v, data in G.edges(data=True):
            pattern = data.get('pattern_type', 'normal')
            if pattern in ['structuring', 'mule_in', 'mule_out']:
                edge_colors.append(self.EDGE_COLORS['suspicious'])
            elif pattern == 'layering':
                edge_colors.append(self.EDGE_COLORS['layering'])
            elif pattern == 'fraud_ring':
                edge_colors.append(self.EDGE_COLORS['mule'])
            else:
                edge_colors.append(self.EDGE_COLORS['normal'])
            
            # Width based on amount
            amount = data.get('weight', 0)
            width = 0.5 + np.log1p(amount) * 0.1
            edge_widths.append(min(width, 5))
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.6,
            arrows=True,
            arrowsize=10,
            connectionstyle="arc3,rad=0.1"
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            edgecolors='white',
            linewidths=0.5
        )
        
        # Draw labels for high-risk nodes
        if show_labels and node_labels:
            nx.draw_networkx_labels(
                G, pos, node_labels, ax=ax,
                font_size=8,
                font_color='white',
                font_weight='bold'
            )
        
        # Highlight suspicious paths
        if highlight_suspicious and self.suspicious_paths:
            for path in self.suspicious_paths[:10]:  # Top 10 paths
                path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                valid_edges = [(u, v) for u, v in path_edges if G.has_edge(u, v)]
                
                if valid_edges:
                    nx.draw_networkx_edges(
                        G, pos, ax=ax,
                        edgelist=valid_edges,
                        edge_color='#E91E63',
                        width=3,
                        alpha=0.8,
                        arrows=True,
                        arrowsize=15,
                        style='dashed'
                    )
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color=self.RISK_COLORS['critical'], label='Critical Risk'),
            mpatches.Patch(color=self.RISK_COLORS['high'], label='High Risk'),
            mpatches.Patch(color=self.RISK_COLORS['medium'], label='Medium Risk'),
            mpatches.Patch(color=self.RISK_COLORS['low'], label='Low Risk'),
            mpatches.Patch(color=self.EDGE_COLORS['suspicious'], label='Suspicious Flow'),
        ]
        
        ax.legend(
            handles=legend_elements,
            loc='upper left',
            facecolor='#16213e',
            edgecolor='#0f3460',
            labelcolor='white',
            fontsize=10
        )
        
        # Title and styling
        ax.set_title(
            title,
            fontsize=20,
            fontweight='bold',
            color='white',
            pad=20
        )
        
        # Add subtitle with stats
        subtitle = (f"Nodes: {G.number_of_nodes()} | "
                   f"Edges: {G.number_of_edges()} | "
                   f"Suspicious Paths: {len(self.suspicious_paths)}")
        ax.text(
            0.5, 0.02, subtitle,
            transform=ax.transAxes,
            fontsize=12,
            color='#888888',
            ha='center'
        )
        
        ax.axis('off')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(
            output_path,
            dpi=150,
            facecolor='#1a1a2e',
            edgecolor='none',
            bbox_inches='tight'
        )
        
        print(f"   └── Saved to: {output_path}")
        
        return fig
    
    def visualize_communities(
        self,
        output_path: str = 'communities_graph.png',
        figsize: Tuple[int, int] = (16, 12)
    ) -> plt.Figure:
        """
        Visualize network communities with distinct colors.
        """
        print("\n🎨 Creating Community Visualization...")
        
        G = self.graph
        
        # Generate colors for communities
        n_communities = len(self.communities)
        cmap = plt.cm.get_cmap('tab20', n_communities)
        
        fig, ax = plt.subplots(figsize=figsize, facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        
        # Use community-aware layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Draw communities
        for comm_id, nodes in self.communities.items():
            if not nodes:
                continue
            
            subgraph = G.subgraph(nodes)
            color = cmap(comm_id / max(n_communities - 1, 1))
            
            nx.draw_networkx_nodes(
                subgraph, pos, ax=ax,
                node_color=[color],
                node_size=100,
                alpha=0.8
            )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color='#444444',
            width=0.5,
            alpha=0.3,
            arrows=True,
            arrowsize=8
        )
        
        ax.set_title(
            f"Transaction Network Communities ({n_communities} clusters)",
            fontsize=16,
            fontweight='bold',
            color='white',
            pad=20
        )
        
        ax.axis('off')
        plt.tight_layout()
        
        plt.savefig(
            output_path,
            dpi=150,
            facecolor='#1a1a2e',
            edgecolor='none',
            bbox_inches='tight'
        )
        
        print(f"   └── Saved to: {output_path}")
        
        return fig
    
    def visualize_suspicious_cluster(
        self,
        account_ids: List[str],
        output_path: str = 'suspicious_cluster.png',
        figsize: Tuple[int, int] = (14, 10),
        depth: int = 2
    ) -> plt.Figure:
        """
        Visualize a specific cluster of suspicious accounts.
        
        Shows the accounts and their immediate network.
        """
        print(f"\n🎨 Visualizing Suspicious Cluster ({len(account_ids)} accounts)...")
        
        G = self.graph
        
        # Get subgraph including neighbors
        nodes_to_include = set(account_ids)
        
        for _ in range(depth):
            new_nodes = set()
            for node in nodes_to_include:
                if node in G:
                    new_nodes.update(G.predecessors(node))
                    new_nodes.update(G.successors(node))
            nodes_to_include.update(new_nodes)
        
        subgraph = G.subgraph(nodes_to_include).copy()
        
        fig, ax = plt.subplots(figsize=figsize, facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        
        pos = nx.spring_layout(subgraph, k=3, iterations=50, seed=42)
        
        # Node colors - highlight original suspicious accounts
        node_colors = []
        node_sizes = []
        
        for node in subgraph.nodes():
            if node in account_ids:
                node_colors.append('#E53935')  # Red for suspicious
                node_sizes.append(400)
            else:
                node_colors.append('#4CAF50')  # Green for neighbors
                node_sizes.append(200)
        
        # Draw
        nx.draw_networkx_edges(
            subgraph, pos, ax=ax,
            edge_color='#666666',
            width=1.5,
            alpha=0.6,
            arrows=True,
            arrowsize=12
        )
        
        nx.draw_networkx_nodes(
            subgraph, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            edgecolors='white',
            linewidths=1
        )
        
        # Labels
        labels = {n: n[:8] for n in subgraph.nodes()}
        nx.draw_networkx_labels(
            subgraph, pos, labels, ax=ax,
            font_size=8,
            font_color='white'
        )
        
        ax.set_title(
            f"Suspicious Account Cluster Analysis",
            fontsize=16,
            fontweight='bold',
            color='white',
            pad=20
        )
        
        ax.axis('off')
        plt.tight_layout()
        
        plt.savefig(
            output_path,
            dpi=150,
            facecolor='#1a1a2e',
            edgecolor='none',
            bbox_inches='tight'
        )
        
        print(f"   └── Saved to: {output_path}")
        
        return fig
    
    def export_graph_data(self, output_dir: str) -> None:
        """
        Export graph data for further analysis.
        
        Creates:
        - nodes.csv: Node list with attributes
        - edges.csv: Edge list with weights
        - network.gexf: Graph file for Gephi
        """
        print("\n📁 Exporting Graph Data...")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        G = self.graph
        
        # Export nodes
        nodes_data = []
        for node in G.nodes(data=True):
            node_id, attrs = node
            metrics = self.node_metrics.get(node_id, {})
            
            nodes_data.append({
                'account_id': node_id,
                'risk_score': attrs.get('risk_score', 0),
                'risk_category': attrs.get('risk_category', 'unknown'),
                'in_degree': G.in_degree(node_id),
                'out_degree': G.out_degree(node_id),
                'pagerank': getattr(metrics, 'pagerank', 0),
                'betweenness': getattr(metrics, 'betweenness', 0),
                'community_id': getattr(metrics, 'community_id', 0)
            })
        
        nodes_df = pd.DataFrame(nodes_data)
        nodes_df.to_csv(f"{output_dir}/graph_nodes.csv", index=False)
        
        # Export edges
        edges_data = []
        for u, v, data in G.edges(data=True):
            edges_data.append({
                'source': u,
                'target': v,
                'weight': data.get('weight', 0),
                'tx_count': data.get('tx_count', 0),
                'pattern_type': data.get('pattern_type', 'normal')
            })
        
        edges_df = pd.DataFrame(edges_data)
        edges_df.to_csv(f"{output_dir}/graph_edges.csv", index=False)
        
        # Export GEXF for Gephi
        nx.write_gexf(G, f"{output_dir}/network.gexf")
        
        print(f"   ├── Nodes: {output_dir}/graph_nodes.csv")
        print(f"   ├── Edges: {output_dir}/graph_edges.csv")
        print(f"   └── GEXF: {output_dir}/network.gexf")


def main():
    """Build and visualize transaction network."""
    # Load data
    transactions = pd.read_csv('/Users/fawazahmad/Desktop/AML/aml-project/data/transactions.csv')
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
    
    try:
        risk_scores = pd.read_csv('/Users/fawazahmad/Desktop/AML/aml-project/data/risk_scores.csv')
    except FileNotFoundError:
        risk_scores = None
    
    # Build graph
    builder = TransactionGraphBuilder()
    G = builder.build_graph(transactions, risk_scores)
    
    # Calculate metrics
    metrics_df = builder.calculate_network_metrics()
    metrics_df.to_csv('/Users/fawazahmad/Desktop/AML/aml-project/data/network_metrics.csv', index=False)
    
    # Find suspicious paths
    builder.find_suspicious_paths()
    
    # Create visualizations
    builder.visualize_network(
        output_path='/Users/fawazahmad/Desktop/AML/aml-project/data/network_graph.png',
        title="AML Suspicious Transaction Network"
    )
    
    builder.visualize_communities(
        output_path='/Users/fawazahmad/Desktop/AML/aml-project/data/communities_graph.png'
    )
    
    # Export graph data
    builder.export_graph_data('/Users/fawazahmad/Desktop/AML/aml-project/data')
    
    print("\n✅ Network analysis complete!")
    
    return builder


if __name__ == '__main__':
    main()


