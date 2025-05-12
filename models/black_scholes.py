import numpy as np
import scipy.stats as si

class BlackScholes:
    """Implementation of the Black-Scholes option pricing model."""
    
    def __init__(self, option_type, s0, k, t, r, sigma, q=0):
        """
        Initialize Black-Scholes model.
        
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
        
        # Common calculations
        self.d1 = (np.log(self.s0 / self.k) + (self.r - self.q + 0.5 * self.sigma**2) * self.t) / (self.sigma * np.sqrt(self.t))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.t)
        
        # Normal cumulative distribution function
        self.N_d1 = si.norm.cdf(self.d1)
        self.N_d2 = si.norm.cdf(self.d2)
        self.N_neg_d1 = si.norm.cdf(-self.d1)
        self.N_neg_d2 = si.norm.cdf(-self.d2)
        
        # Normal probability density function
        self.n_d1 = si.norm.pdf(self.d1)
    
    def price(self):
        """
        Calculate the option price using the Black-Scholes formula.
        
        Returns:
        --------
        float
            Option price
        """
        if self.option_type == 'call':
            return self.s0 * np.exp(-self.q * self.t) * self.N_d1 - self.k * np.exp(-self.r * self.t) * self.N_d2
        else:  # put
            return self.k * np.exp(-self.r * self.t) * self.N_neg_d2 - self.s0 * np.exp(-self.q * self.t) * self.N_neg_d1
    
    def delta(self):
        """
        Calculate the delta of the option.
        
        Returns:
        --------
        float
            Option delta
        """
        if self.option_type == 'call':
            return np.exp(-self.q * self.t) * self.N_d1
        else:  # put
            return np.exp(-self.q * self.t) * (self.N_d1 - 1)
    
    def gamma(self):
        """
        Calculate the gamma of the option.
        
        Returns:
        --------
        float
            Option gamma
        """
        return np.exp(-self.q * self.t) * self.n_d1 / (self.s0 * self.sigma * np.sqrt(self.t))
    
    def vega(self):
        """
        Calculate the vega of the option.
        
        Returns:
        --------
        float
            Option vega (for 1% change in volatility)
        """
        return 0.01 * self.s0 * np.exp(-self.q * self.t) * np.sqrt(self.t) * self.n_d1
    
    def theta(self):
        """
        Calculate the theta of the option.
        
        Returns:
        --------
        float
            Option theta (for 1 day change, dividing by 365)
        """
        if self.option_type == 'call':
            theta = -np.exp(-self.q * self.t) * self.s0 * self.n_d1 * self.sigma / (2 * np.sqrt(self.t)) \
                   - self.r * self.k * np.exp(-self.r * self.t) * self.N_d2 \
                   + self.q * self.s0 * np.exp(-self.q * self.t) * self.N_d1
        else:  # put
            theta = -np.exp(-self.q * self.t) * self.s0 * self.n_d1 * self.sigma / (2 * np.sqrt(self.t)) \
                   + self.r * self.k * np.exp(-self.r * self.t) * self.N_neg_d2 \
                   - self.q * self.s0 * np.exp(-self.q * self.t) * self.N_neg_d1
        
        # Convert to daily theta
        return theta / 365
    
    def rho(self):
        """
        Calculate the rho of the option.
        
        Returns:
        --------
        float
            Option rho (for 1% change in interest rate)
        """
        if self.option_type == 'call':
            return 0.01 * self.k * self.t * np.exp(-self.r * self.t) * self.N_d2
        else:  # put
            return -0.01 * self.k * self.t * np.exp(-self.r * self.t) * self.N_neg_d2