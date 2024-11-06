import networkx as nx
import numpy as np
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from scipy.io import mmread
import scipy.sparse as sp


class TwitterInfluenceAnalyzer:
    def __init__(self, mtx_file):
        """
        Initialize the analyzer with an MTX file path

        Parameters:
        mtx_file (str): Path to the .mtx file
        """
        self.mtx_file = mtx_file
        self.graph = None
        self.adjacency_matrix = None

    def load_mtx_data(self):
        """
        Load the Twitter network from MTX format
        """
        print("Loading MTX file...")
        try:
            # Read the MTX file using scipy
            self.adjacency_matrix = mmread(self.mtx_file)
            print(f"Matrix shape: {self.adjacency_matrix.shape}")

            # Convert to CSR format for efficient operations
            self.adjacency_matrix = sp.csr_matrix(self.adjacency_matrix)

            # Create NetworkX graph
            print("Converting to NetworkX graph...")
            self.graph = nx.from_scipy_sparse_array(
                self.adjacency_matrix,
                create_using=nx.DiGraph
            )

            print(f"Graph created with {self.graph.number_of_nodes()} nodes and "
                  f"{self.graph.number_of_edges()} edges")

        except Exception as e:
            print(f"Error loading MTX file: {str(e)}")
            raise

    def get_top_influential_users(self, method='outdegree', n=10):
        """
        Find influential users using various metrics

        Parameters:
        method (str): Method to use ('outdegree', 'pagerank', or 'eigenvector')
        n (int): Number of top users to return

        Returns:
        list: Top influential users with their scores
        """
        if method == 'outdegree':
            scores = dict(self.graph.out_degree())
        elif method == 'pagerank':
            scores = nx.pagerank(self.graph)
        elif method == 'eigenvector':
            try:
                scores = nx.eigenvector_centrality(self.graph)
            except:
                print("Eigenvector centrality failed to converge, falling back to out-degree")
                scores = dict(self.graph.out_degree())
        else:
            raise ValueError("Invalid method specified")

        # Sort by score
        top_users = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return top_users

    def analyze_network_stats(self):
        """
        Calculate and return basic network statistics
        """
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'avg_out_degree': np.mean([d for n, d in self.graph.out_degree()]),
            'max_out_degree': max([d for n, d in self.graph.out_degree()]),
            'strongly_connected_components': nx.number_strongly_connected_components(self.graph)
        }
        return stats

    def simulate_influence_spread(self, seed_nodes, prob=0.1, iterations=100):
        """
        Simulate influence spread from seed nodes

        Parameters:
        seed_nodes (list): Initial set of influential nodes
        prob (float): Propagation probability
        iterations (int): Number of Monte Carlo iterations

        Returns:
        float: Average number of influenced nodes
        """
        total_influenced = 0

        for _ in tqdm(range(iterations), desc="Simulating influence"):
            influenced = set(seed_nodes)
            active = set(seed_nodes)

            while active:
                new_active = set()
                for node in active:
                    neighbors = set(self.graph.successors(node)) - influenced
                    for neighbor in neighbors:
                        if np.random.random() < prob:
                            new_active.add(neighbor)
                            influenced.add(neighbor)
                active = new_active

            total_influenced += len(influenced)

        return total_influenced / iterations


def main():
    # File path
    mtx_file = "soc-twitter-follows.mtx"

    # Initialize analyzer
    print("Initializing Twitter network analyzer...")
    analyzer = TwitterInfluenceAnalyzer(mtx_file)

    try:
        # Load the network
        analyzer.load_mtx_data()

        # Calculate and display network statistics
        print("\nCalculating network statistics...")
        stats = analyzer.analyze_network_stats()
        print("\nNetwork Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")

        # Find influential users using different methods
        methods = ['outdegree', 'pagerank']
        results = {}

        print("\nFinding influential users...")
        for method in methods:
            print(f"\nTop 10 users by {method}:")
            top_users = analyzer.get_top_influential_users(method=method, n=10)
            results[method] = top_users
            print(f"\nTop {method} users:")
            for user_id, score in top_users:
                print(f"User {user_id}: {score:.4f}")

        # Simulate influence spread for top users
        print("\nSimulating influence spread...")
        top_users_outdegree = [user[0] for user in results['outdegree'][:5]]
        influence_spread = analyzer.simulate_influence_spread(
            top_users_outdegree,
            prob=0.1,
            iterations=50
        )
        print(f"\nAverage influence spread from top 5 users: {influence_spread:.2f} nodes")

        # Save results to CSV
        print("\nSaving results...")
        results_df = pd.DataFrame({
            'user_id': [user[0] for user in results['outdegree']],
            'outdegree_score': [user[1] for user in results['outdegree']],
            'pagerank_score': [dict(results['pagerank']).get(user[0], 0)
                               for user in results['outdegree']]
        })
        results_df.to_csv('twitter_influential_users.csv', index=False)
        print("Results saved to twitter_influential_users.csv")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()