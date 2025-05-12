import streamlit as st

def format_number(value):
    """
    Format numbers for display.
    
    Parameters:
    -----------
    value : float
        Number to format
    
    Returns:
    --------
    str
        Formatted number
    """
    # Format to 4 decimal places for small numbers, 2 otherwise
    if abs(value) < 0.01:
        return f"{value:.4f}"
    else:
        return f"{value:.2f}"

def create_example_inputs(option_type):
    """
    Create example input parameters.
    
    Parameters:
    -----------
    option_type : str
        Type of option ('european' or 'american')
    
    Returns:
    --------
    dict
        Dictionary of example parameter values
    """
    if option_type == "european":
        return {
            "option_type": "Call",
            "s0": 100.0,
            "k": 100.0,
            "t": 1.0,
            "r": 0.05,
            "sigma": 0.2,
            "q": 0.01
        }
    else:  # american
        return {
            "option_type": "Put",
            "s0": 100.0,
            "k": 100.0,
            "t": 1.0,
            "r": 0.05,
            "sigma": 0.2,
            "q": 0.03
        }

def display_about_section():
    """Display information about the app."""
    st.markdown("<h2 class='section-header'>About the Option Pricer</h2>", unsafe_allow_html=True)
    
    # Display app description
    st.markdown("""
    <div class='about-section'>
        <h3>Overview</h3>
        <p>
            This option pricing application provides tools for pricing European and American options using industry-standard 
            quantitative finance models. It is designed for financial professionals, students, and anyone interested in 
            understanding option pricing mechanics.
        </p>
        
        <h3>Implemented Models</h3>
        <h4>European Options</h4>
        <ul>
            <li>
                <strong>Black-Scholes Model</strong>: The classic analytical solution for European option pricing, 
                incorporating underlying price, strike, time to maturity, risk-free rate, volatility, and dividend yield.
            </li>
            <li>
                <strong>Monte Carlo Simulation</strong>: A numerical method that simulates thousands of possible price paths 
                to estimate the option price.
            </li>
        </ul>
        
        <h4>American Options</h4>
        <ul>
            <li>
                <strong>Binomial Tree (Cox-Ross-Rubinstein)</strong>: A lattice-based model that divides time to maturity 
                into discrete periods and models the evolution of the underlying asset price as a binomial process.
            </li>
            <li>
                <strong>Least Squares Monte Carlo (Longstaff-Schwartz)</strong>: A regression-based approach that uses 
                Monte Carlo simulation with backward induction to determine optimal exercise policy.
            </li>
        </ul>
        
        <h3>Mathematical Background</h3>
        <h4>Black-Scholes Formula</h4>
        <p>
            For a European call option:
        </p>
        <p class='formula'>
            C = S₀e⁻ᵈᵗN(d₁) - Ke⁻ʳᵗN(d₂)
        </p>
        <p>
            For a European put option:
        </p>
        <p class='formula'>
            P = Ke⁻ʳᵗN(-d₂) - S₀e⁻ᵈᵗN(-d₁)
        </p>
        <p>
            where:
        </p>
        <p class='formula'>
            d₁ = [ln(S₀/K) + (r - q + σ²/2)t] / (σ√t)
        </p>
        <p class='formula'>
            d₂ = d₁ - σ√t
        </p>
        
        <h4>Greeks Calculations</h4>
        <p>The Greeks measure the sensitivity of option prices to various factors:</p>
        <ul>
            <li><strong>Delta (Δ)</strong>: Sensitivity to underlying price</li>
            <li><strong>Gamma (Γ)</strong>: Rate of change of Delta</li>
            <li><strong>Theta (Θ)</strong>: Sensitivity to time decay</li>
            <li><strong>Vega (ν)</strong>: Sensitivity to volatility</li>
            <li><strong>Rho (ρ)</strong>: Sensitivity to interest rate</li>
        </ul>
        
        <h3>References</h3>
        <ul>
            <li>Hull, J. C. (2018). <em>Options, Futures, and Other Derivatives</em>. Pearson.</li>
            <li>Wilmott, P. (2006). <em>Paul Wilmott On Quantitative Finance</em>. Wiley.</li>
            <li>Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options by simulation: A simple least-squares approach. <em>The Review of Financial Studies</em>, 14(1), 113-147.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Display example usage section
    st.markdown("### Example Usage", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='example-usage'>
        <p>
            <strong>For European Options:</strong>
        </p>
        <ol>
            <li>Select 'European Options' in the sidebar</li>
            <li>Choose option type (Call/Put)</li>
            <li>Enter parameters (or use example values)</li>
            <li>View the calculated price and Greeks</li>
            <li>Analyze the sensitivity charts</li>
            <li>Download results if needed</li>
        </ol>
        
        <p>
            <strong>For American Options:</strong>
        </p>
        <ol>
            <li>Select 'American Options' in the sidebar</li>
            <li>Choose option type (Call/Put)</li>
            <li>Enter parameters (or use example values)</li>
            <li>Adjust the number of time steps for the binomial tree</li>
            <li>View the calculated price and early exercise premium</li>
            <li>Explore the visualizations to understand exercise boundaries</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Display disclaimer
    st.markdown("""
    <div class='disclaimer'>
        <h3>Disclaimer</h3>
        <p>
            This application is provided for educational and informational purposes only. It is not intended as financial 
            advice or as a recommendation to trade options. Options trading involves significant risk and is not suitable 
            for all investors. The calculations provided by this tool are theoretical and based on mathematical models that 
            make certain assumptions about market conditions.
        </p>
        <p>
            Real market prices may differ due to factors not accounted for in these models, such as liquidity, transaction 
            costs, and market inefficiencies. Always consult with a qualified financial advisor before making investment 
            decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)