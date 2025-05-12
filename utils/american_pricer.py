import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from models.binomial_tree import BinomialTree
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

class AmericanPricer:
    """Class for pricing American options using various methods."""
    
    def __init__(self, option_type, s0, k, t, r, sigma, q=0):
        """
        Initialize the American option pricer.
        
        Parameters:
        -----------
        option_type : str
            Type of option ('call' or 'put')
        s0 : float
            Initial stock price
        k : float
            Strike price
        t : float
            Time to maturity in years
        r : float
            Risk-free interest rate (annualized)
        sigma : float
            Volatility (annualized)
        q : float, optional
            Dividend yield (annualized), default is 0
        """
        self.option_type = option_type
        self.s0 = s0
        self.k = k
        self.t = t
        self.r = r
        self.sigma = sigma
        self.q = q
        self.binomial_model = BinomialTree(self.option_type, self.s0, self.k, self.t, self.r, self.sigma, self.q)
    
    def binomial_tree_price(self, n_steps=50):
        """
        Calculate option price using the Binomial Tree (Cox-Ross-Rubinstein) model.
        
        Parameters:
        -----------
        n_steps : int
            Number of time steps in the binomial tree
            
        Returns:
        --------
        tuple
            (option_price, tree_data, exercise_boundary)
            - option_price: float
            - tree_data: dict containing stock and option price trees
            - exercise_boundary: dict containing the early exercise boundary
        """
        return self.binomial_model.price(n_steps)
    
    def least_squares_mc_price(self, num_simulations=10000, n_steps=50):
        """
        Calculate option price using the Least Squares Monte Carlo (LSM) method.
        
        Parameters:
        -----------
        num_simulations : int
            Number of Monte Carlo simulations
        n_steps : int
            Number of time steps
            
        Returns:
        --------
        tuple
            (option_price, paths, exercise_times)
            - option_price: float
            - paths: numpy array of simulated price paths
            - exercise_times: numpy array of optimal exercise times for each path
        """
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Time step
        dt = self.t / n_steps
        
        # Generate stock price paths
        paths = np.zeros((num_simulations, n_steps + 1))
        paths[:, 0] = self.s0
        
        # Simulate geometric Brownian motion
        for i in range(1, n_steps + 1):
            z = np.random.standard_normal(num_simulations)
            paths[:, i] = paths[:, i-1] * np.exp((self.r - self.q - 0.5 * self.sigma**2) * dt + 
                                               self.sigma * np.sqrt(dt) * z)
        
        # Initialize payoff matrix
        if self.option_type == 'call':
            payoffs = np.maximum(paths - self.k, 0)
        else:
            payoffs = np.maximum(self.k - paths, 0)
        
        # Initialize matrix to store continuation values
        continuation_values = np.zeros_like(paths)
        
        # Initialize exercise decisions (0: hold, 1: exercise)
        exercise_decisions = np.zeros((num_simulations, n_steps + 1))
        
        # Backward induction through the tree
        # Start from maturity, where we always get the payoff
        continuation_values[:, -1] = payoffs[:, -1]
        exercise_decisions[:, -1] = 1  # Always exercise at maturity if in-the-money
        
        # Work backward through time
        for i in range(n_steps - 1, 0, -1):
            # Identify in-the-money paths
            itm_idx = payoffs[:, i] > 0
            
            if np.sum(itm_idx) > 0:
                # Get in-the-money paths
                itm_paths = paths[itm_idx, i]
                
                # Future discounted cashflows
                future_cf = continuation_values[itm_idx, i+1] * np.exp(-self.r * dt)
                
                # Fit regression model
                x = itm_paths
                y = future_cf
                
                # Use polynomial basis functions (degree 2)
                reg_coef = np.polyfit(x, y, 2)
                expected_cf = np.polyval(reg_coef, itm_paths)
                
                # Determine optimal exercise decision
                immediate_payoff = payoffs[itm_idx, i]
                
                # Update continuation values based on regression
                continuation_values[itm_idx, i] = expected_cf
                
                # Decide whether to exercise
                exercise_idx = immediate_payoff > expected_cf
                
                # For paths where immediate exercise is optimal
                exercise_decisions[itm_idx, i][exercise_idx] = 1
                
                # Update continuation values for exercised paths
                continuation_values[itm_idx, i][exercise_idx] = immediate_payoff[exercise_idx]
            
            # For out-of-the-money paths, continuation value is discounted future value
            otm_idx = ~itm_idx
            continuation_values[otm_idx, i] = continuation_values[otm_idx, i+1] * np.exp(-self.r * dt)
        
        # Determine the optimal exercise times for each path
        exercise_times = np.zeros(num_simulations)
        for j in range(num_simulations):
            # Find the first time we should exercise
            ex_indices = np.where(exercise_decisions[j, :] == 1)[0]
            if len(ex_indices) > 0:
                exercise_times[j] = ex_indices[0]
            else:
                exercise_times[j] = n_steps  # Exercise at maturity
        
        # Calculate the option price
        # For each path, we use the payoff at the optimal exercise time
        option_values = np.zeros(num_simulations)
        for j in range(num_simulations):
            if exercise_times[j] < n_steps:  # If early exercise is optimal
                time_idx = int(exercise_times[j])
                option_values[j] = payoffs[j, time_idx] * np.exp(-self.r * time_idx * dt)
            else:  # Otherwise, use payoff at maturity
                option_values[j] = payoffs[j, -1] * np.exp(-self.r * self.t)
        
        # The option price is the average of all path values
        option_price = np.mean(option_values)
        
        return option_price, paths, exercise_times
    
    def plot_exercise_boundary(self, exercise_boundary):
        """
        Plot the early exercise boundary from a binomial tree.
        
        Parameters:
        -----------
        exercise_boundary : dict
            Dictionary with time points and corresponding stock prices
            at which early exercise is optimal
        """
        # Extract data
        times = np.array(list(exercise_boundary.keys()))
        boundaries = np.array(list(exercise_boundary.values()))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        ax.plot(times, boundaries, 'r-o', linewidth=2, markersize=4)
        ax.axhline(y=self.k, color='k', linestyle='--', alpha=0.7, label=f"Strike Price (K = {self.k})")
        
        # Shade regions
        if self.option_type == 'put':
            # For put options, exercise region is below the boundary
            ax.fill_between(times, 0, boundaries, alpha=0.2, color='red', label="Exercise Region")
            ax.fill_between(times, boundaries, max(boundaries) * 1.5, alpha=0.2, color='green', label="Continuation Region")
        else:
            # For call options, exercise region is above the boundary
            ax.fill_between(times, boundaries, max(boundaries) * 1.5, alpha=0.2, color='red', label="Exercise Region")
            ax.fill_between(times, 0, boundaries, alpha=0.2, color='green', label="Continuation Region")
        
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Stock Price")
        ax.set_title(f"Early Exercise Boundary for {self.option_type.capitalize()} Option")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    def plot_binomial_tree(self, tree_data, max_steps=5):
        """
        Visualize the binomial tree (for a limited number of steps).
        
        Parameters:
        -----------
        tree_data : dict
            Dictionary containing stock and option price trees
        max_steps : int
            Maximum number of time steps to display
        """
        # Extract data
        stock_tree = tree_data['stock_tree']
        option_tree = tree_data['option_tree']
        exercise_tree = tree_data['exercise_tree']
        
        # Limit to max_steps
        n_steps = min(max_steps, len(stock_tree[0]) - 1)
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges to the graph
        for i in range(n_steps + 1):
            for j in range(i + 1):
                # Node ID and position
                node_id = f"{i},{j}"
                pos_x = i
                pos_y = j - i/2  # Center the tree
                
                # Stock and option prices at this node
                stock_price = stock_tree[j][i]
                option_price = option_tree[j][i]
                
                # Node attributes
                G.add_node(
                    node_id, 
                    pos=(pos_x, pos_y),
                    stock_price=stock_price,
                    option_price=option_price,
                    exercise=exercise_tree[j][i] if i < len(exercise_tree[0]) else False
                )
                
                # Add edges to predecessor nodes (if not at t=0)
                if i > 0:
                    # Down move
                    G.add_edge(f"{i-1},{j-1}", node_id)
                    # Up move
                    if j < i:
                        G.add_edge(f"{i-1},{j}", node_id)
        
        # Get node positions
        pos = nx.get_node_attributes(G, 'pos')
        
        # Get node colors based on exercise decision
        node_colors = []
        for node in G.nodes():
            if G.nodes[node]['exercise']:
                node_colors.append('lightcoral')  # Exercise nodes
            else:
                node_colors.append('lightblue')   # Continuation nodes
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw the graph
        nx.draw(
            G, 
            pos=pos,
            with_labels=False,
            node_color=node_colors,
            node_size=500,
            arrows=True,
            arrowsize=20,
            ax=ax
        )
        
        # Add labels to nodes
        node_labels = {}
        for node in G.nodes():
            stock_price = G.nodes[node]['stock_price']
            option_price = G.nodes[node]['option_price']
            node_labels[node] = f"S: {stock_price:.2f}\nO: {option_price:.2f}"
        
        nx.draw_networkx_labels(
            G, 
            pos=pos,
            labels=node_labels,
            font_size=8,
            font_color='black'
        )
        
        # Add legend
        import matplotlib.patches as mpatches
        blue_patch = mpatches.Patch(color='lightblue', label='Continuation')
        red_patch = mpatches.Patch(color='lightcoral', label='Exercise')
        ax.legend(handles=[blue_patch, red_patch])
        
        ax.set_title(f"Binomial Tree for {self.option_type.capitalize()} Option (First {n_steps} steps)")
        ax.axis('off')
        
        st.pyplot(fig)
    
    def plot_lsmc_results(self, paths, exercise_times):
        """
        Visualize the Least Squares Monte Carlo results.
        
        Parameters:
        -----------
        paths : numpy.ndarray
            Array of simulated price paths
        exercise_times : numpy.ndarray
            Array of optimal exercise times for each path
        """
        # Select a subset of paths for visualization
        num_paths_to_show = min(50, paths.shape[0])
        path_indices = np.random.choice(paths.shape[0], num_paths_to_show, replace=False)
        
        # Create time array
        n_steps = paths.shape[1] - 1
        times = np.linspace(0, self.t, n_steps + 1)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a custom colormap for paths
        cmap = LinearSegmentedColormap.from_list("custom_cmap", ["blue", "green", "red"])
        
        # Plot selected paths
        for i, idx in enumerate(path_indices):
            # Get the path and exercise time
            path = paths[idx]
            ex_time = exercise_times[idx]
            
            # Plot the path
            ax.plot(times, path, alpha=0.3, color=cmap(i / num_paths_to_show))
            
            # Mark the exercise point if early exercise occurred
            if ex_time < n_steps:
                ex_time_idx = int(ex_time)
                ax.scatter(times[ex_time_idx], path[ex_time_idx], color='red', s=50, zorder=3)
        
        # Add strike price line
        ax.axhline(y=self.k, color='k', linestyle='--', alpha=0.7, label=f"Strike Price (K = {self.k})")
        
        # Add scatter for early exercise points
        ax.scatter([], [], color='red', s=50, label='Early Exercise Points')
        
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Stock Price")
        ax.set_title(f"Monte Carlo Paths for {self.option_type.capitalize()} Option")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    def download_results(self, model_type="binomial_tree"):
        """
        Create a CSV file with option pricing results.
        
        Parameters:
        -----------
        model_type : str
            Type of pricing model used (for file naming)
        """
        # Calculate option price
        if model_type == "binomial_tree":
            price, _, _ = self.binomial_tree_price()
            method = "Binomial Tree"
        else:
            price, _, _ = self.least_squares_mc_price()
            method = "Least Squares Monte Carlo"
        
        # Create a dictionary with all parameters and results
        data = {
            "Parameter": [
                "Option Type", "Underlying Price (S₀)", "Strike Price (K)", 
                "Time to Maturity (T)", "Risk-free Rate (r)", "Volatility (σ)", 
                "Dividend Yield (q)", "Pricing Method", "Option Price"
            ],
            "Value": [
                self.option_type.capitalize(), self.s0, self.k, self.t, 
                self.r, self.sigma, self.q, method, price
            ]
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Create CSV string
        csv = df.to_csv(index=False)
        
        # Create download button
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"american_option_{model_type}_results.csv",
            mime="text/csv"
        )