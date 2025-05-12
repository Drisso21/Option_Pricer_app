import numpy as np

class BinomialTree:
    """Implementation of the Binomial Tree model for option pricing."""
    
    def __init__(self, option_type, s0, k, t, r, sigma, q=0):
        """
        Initialize Binomial Tree model.
        
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
    
    def price(self, n_steps=50):
        """
        Calculate the option price using the Binomial Tree (Cox-Ross-Rubinstein) model.
        
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
        # Time increment
        dt = self.t / n_steps
        
        # Calculate up and down factors
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        
        # Risk-neutral probability
        a = np.exp((self.r - self.q) * dt)
        p = (a - d) / (u - d)
        
        # Discount factor
        discount = np.exp(-self.r * dt)
        
        # Initialize stock price tree
        stock_tree = [[0.0 for _ in range(n_steps + 1)] for _ in range(n_steps + 1)]
        
        # Populate stock price tree
        for i in range(n_steps + 1):
            for j in range(i + 1):
                stock_tree[j][i] = self.s0 * (u ** (i - j)) * (d ** j)
        
        # Initialize option value tree
        option_tree = [[0.0 for _ in range(n_steps + 1)] for _ in range(n_steps + 1)]
        
        # Initialize exercise decision tree (True = exercise, False = continue)
        exercise_tree = [[False for _ in range(n_steps + 1)] for _ in range(n_steps + 1)]
        
        # Initialize early exercise boundary
        exercise_boundary = {}
        
        # Populate option value at maturity (terminal nodes)
        for j in range(n_steps + 1):
            if self.option_type == 'call':
                option_tree[j][n_steps] = max(0, stock_tree[j][n_steps] - self.k)
            else:  # put
                option_tree[j][n_steps] = max(0, self.k - stock_tree[j][n_steps])
            
            # At maturity, exercise if in-the-money
            exercise_tree[j][n_steps] = option_tree[j][n_steps] > 0
        
        # Backward induction through the tree
        for i in range(n_steps - 1, -1, -1):
            # For each node at this time step
            for j in range(i + 1):
                # Calculate expected option value (continuation value)
                continuation_value = discount * (p * option_tree[j][i+1] + (1 - p) * option_tree[j+1][i+1])
                
                # Calculate intrinsic value
                if self.option_type == 'call':
                    intrinsic_value = max(0, stock_tree[j][i] - self.k)
                else:  # put
                    intrinsic_value = max(0, self.k - stock_tree[j][i])
                
                # For American options, take max of continuation and intrinsic
                option_tree[j][i] = max(continuation_value, intrinsic_value)
                
                # Record exercise decision
                exercise_tree[j][i] = option_tree[j][i] == intrinsic_value and intrinsic_value > 0
                
                # Record early exercise boundary for each time step
                time_point = i * dt
                stock_price = stock_tree[j][i]
                
                # For call options, we're looking for the lowest stock price
                # where exercise is optimal
                if self.option_type == 'call' and exercise_tree[j][i]:
                    if time_point not in exercise_boundary or stock_price < exercise_boundary[time_point]:
                        exercise_boundary[time_point] = stock_price
                
                # For put options, we're looking for the highest stock price
                # where exercise is optimal
                elif self.option_type == 'put' and exercise_tree[j][i]:
                    if time_point not in exercise_boundary or stock_price > exercise_boundary[time_point]:
                        exercise_boundary[time_point] = stock_price
        
        # Option price is at the root of the tree
        option_price = option_tree[0][0]
        
        # Return option price, tree data, and exercise boundary
        tree_data = {
            'stock_tree': stock_tree,
            'option_tree': option_tree,
            'exercise_tree': exercise_tree
        }
        
        return option_price, tree_data, exercise_boundary