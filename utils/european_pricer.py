import numpy as np
import scipy.stats as si
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from models.black_scholes import BlackScholes

class EuropeanPricer:
    """Class for pricing European options and calculating Greeks."""
    
    def __init__(self, option_type, s0, k, t, r, sigma, q=0):
        """
        Initialize the European option pricer.
        
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
        self.bs_model = BlackScholes(self.option_type, self.s0, self.k, self.t, self.r, self.sigma, self.q)
    
    def black_scholes_price(self):
        """Calculate option price using Black-Scholes model."""
        return self.bs_model.price()
    
    def delta(self):
        """Calculate option delta."""
        return self.bs_model.delta()
    
    def gamma(self):
        """Calculate option gamma."""
        return self.bs_model.gamma()
    
    def vega(self):
        """Calculate option vega."""
        return self.bs_model.vega()
    
    def theta(self):
        """Calculate option theta."""
        return self.bs_model.theta()
    
    def rho(self):
        """Calculate option rho."""
        return self.bs_model.rho()
    
    def monte_carlo_price(self, num_simulations=10000):
        """
        Calculate option price using Monte Carlo simulation.
        
        Parameters:
        -----------
        num_simulations : int
            Number of simulations to run
            
        Returns:
        --------
        float
            Option price
        """
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate random paths
        dt = self.t
        z = np.random.standard_normal(num_simulations)
        s_t = self.s0 * np.exp((self.r - self.q - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z)
        
        # Calculate payoffs
        if self.option_type == 'call':
            payoffs = np.maximum(s_t - self.k, 0)
        else:
            payoffs = np.maximum(self.k - s_t, 0)
        
        # Discount payoffs to present value
        option_price = np.exp(-self.r * self.t) * np.mean(payoffs)
        
        return option_price
    
    def monte_carlo_greeks(self, num_simulations=10000):
        """
        Calculate option Greeks using Monte Carlo simulation with finite differences.
        
        Parameters:
        -----------
        num_simulations : int
            Number of simulations to run
            
        Returns:
        --------
        tuple
            (delta, gamma, theta, vega, rho)
        """
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Parameters for finite difference
        h_s = self.s0 * 0.01  # 1% of stock price
        h_t = 1/365           # 1 day
        h_sigma = 0.01        # 1% volatility
        h_r = 0.0001          # 1 basis point
        
        # Base price
        base_price = self.monte_carlo_price(num_simulations)
        
        # Delta
        pricer_up = EuropeanPricer(self.option_type, self.s0 + h_s, self.k, self.t, self.r, self.sigma, self.q)
        price_up = pricer_up.monte_carlo_price(num_simulations)
        delta = (price_up - base_price) / h_s
        
        # Gamma
        pricer_down = EuropeanPricer(self.option_type, self.s0 - h_s, self.k, self.t, self.r, self.sigma, self.q)
        price_down = pricer_down.monte_carlo_price(num_simulations)
        gamma = (price_up - 2 * base_price + price_down) / (h_s**2)
        
        # Theta
        pricer_t_down = EuropeanPricer(self.option_type, self.s0, self.k, self.t - h_t, self.r, self.sigma, self.q)
        price_t_down = pricer_t_down.monte_carlo_price(num_simulations)
        theta = (price_t_down - base_price) / h_t
        
        # Vega
        pricer_sigma_up = EuropeanPricer(self.option_type, self.s0, self.k, self.t, self.r, self.sigma + h_sigma, self.q)
        price_sigma_up = pricer_sigma_up.monte_carlo_price(num_simulations)
        vega = (price_sigma_up - base_price) / h_sigma
        
        # Rho
        pricer_r_up = EuropeanPricer(self.option_type, self.s0, self.k, self.t, self.r + h_r, self.sigma, self.q)
        price_r_up = pricer_r_up.monte_carlo_price(num_simulations)
        rho = (price_r_up - base_price) / h_r
        
        return delta, gamma, theta, vega, rho
    
    def plot_price_vs_strike(self):
        """Plot option price vs strike price."""
        strikes = np.linspace(self.k * 0.5, self.k * 1.5, 50)
        prices = []
        
        for strike in strikes:
            pricer = EuropeanPricer(self.option_type, self.s0, strike, self.t, self.r, self.sigma, self.q)
            prices.append(pricer.black_scholes_price())
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        ax.plot(strikes, prices, 'b-', linewidth=2)
        ax.axvline(x=self.s0, color='r', linestyle='--', alpha=0.7, label=f"Current Price (S₀ = {self.s0})")
        
        ax.set_xlabel("Strike Price (K)")
        ax.set_ylabel("Option Price")
        ax.set_title(f"{self.option_type.capitalize()} Option Price vs Strike Price")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add shading for in-the-money and out-of-the-money regions
        if self.option_type == 'call':
            ax.axvspan(0, self.s0, alpha=0.2, color='green', label="In-the-money (Call)")
            ax.axvspan(self.s0, max(strikes), alpha=0.2, color='red', label="Out-of-the-money (Call)")
        else:
            ax.axvspan(0, self.s0, alpha=0.2, color='red', label="Out-of-the-money (Put)")
            ax.axvspan(self.s0, max(strikes), alpha=0.2, color='green', label="In-the-money (Put)")
        
        ax.legend()
        
        st.pyplot(fig)
    
    def plot_price_vs_volatility(self):
        """Plot option price vs volatility."""
        vols = np.linspace(max(0.05, self.sigma * 0.5), self.sigma * 1.5, 50)
        prices = []
        
        for vol in vols:
            pricer = EuropeanPricer(self.option_type, self.s0, self.k, self.t, self.r, vol, self.q)
            prices.append(pricer.black_scholes_price())
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        ax.plot(vols, prices, 'b-', linewidth=2)
        ax.axvline(x=self.sigma, color='r', linestyle='--', alpha=0.7, label=f"Current Volatility (σ = {self.sigma:.2f})")
        
        ax.set_xlabel("Volatility (σ)")
        ax.set_ylabel("Option Price")
        ax.set_title(f"{self.option_type.capitalize()} Option Price vs Volatility")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    def plot_price_vs_time(self):
        """Plot option price vs time to maturity."""
        times = np.linspace(max(0.05, self.t * 0.1), self.t * 1.5, 50)
        prices = []
        
        for t in times:
            pricer = EuropeanPricer(self.option_type, self.s0, self.k, t, self.r, self.sigma, self.q)
            prices.append(pricer.black_scholes_price())
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        ax.plot(times, prices, 'b-', linewidth=2)
        ax.axvline(x=self.t, color='r', linestyle='--', alpha=0.7, label=f"Current Time to Maturity (T = {self.t:.2f} years)")
        
        ax.set_xlabel("Time to Maturity (years)")
        ax.set_ylabel("Option Price")
        ax.set_title(f"{self.option_type.capitalize()} Option Price vs Time to Maturity")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    def download_results(self):
        """Create a CSV file with option pricing results."""
        # Calculate option price and Greeks
        price = self.black_scholes_price()
        delta = self.delta()
        gamma = self.gamma()
        theta = self.theta()
        vega = self.vega()
        rho = self.rho()
        
        # Create a dictionary with all parameters and results
        data = {
            "Parameter": [
                "Option Type", "Underlying Price (S₀)", "Strike Price (K)", 
                "Time to Maturity (T)", "Risk-free Rate (r)", "Volatility (σ)", 
                "Dividend Yield (q)", "Option Price", "Delta (Δ)", 
                "Gamma (Γ)", "Theta (Θ)", "Vega (ν)", "Rho (ρ)"
            ],
            "Value": [
                self.option_type.capitalize(), self.s0, self.k, self.t, 
                self.r, self.sigma, self.q, price, delta, gamma, theta, vega, rho
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
            file_name="european_option_results.csv",
            mime="text/csv"
        )