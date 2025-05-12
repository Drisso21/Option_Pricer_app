import streamlit as st
import numpy as np
from utils.european_pricer import EuropeanPricer
from utils.american_pricer import AmericanPricer
from utils.helpers import format_number, create_example_inputs, display_about_section
import os

# Set page configuration
st.set_page_config(
    page_title="Option Pricer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open(os.path.join("assets", "style.css")) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize session state if not already done
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "European Options"

# App header
st.markdown("<h1 class='app-header'>Quantitative Option Pricer</h1>", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.markdown("<h2 class='sidebar-header'>Navigation</h2>", unsafe_allow_html=True)
tabs = ["European Options", "American Options", "About"]
selected_tab = st.sidebar.radio("Select Option Type", tabs, index=tabs.index(st.session_state.current_tab))
st.session_state.current_tab = selected_tab

# Sidebar - Display author info at bottom
st.sidebar.markdown("---")
st.sidebar.markdown("<div class='sidebar-footer'>Developed By ISFA2023 </div>", unsafe_allow_html=True)

# Main content
if selected_tab == "European Options":
    st.markdown("<h2 class='section-header'>European Option Pricing</h2>", unsafe_allow_html=True)
    
    # Create two columns for inputs and results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h3 class='subsection-header'>Input Parameters</h3>", unsafe_allow_html=True)
        
        # Option example presets
        use_example = st.checkbox("Use example values", value=False)
        if use_example:
            example_values = create_example_inputs("european")
            
        # Option type selection
        option_type = st.selectbox(
            "Option Type",
            ["Call", "Put"],
            index=0 if not use_example else 0 if example_values["option_type"] == "Call" else 1
        )
        
        # Option parameters with tooltips
        with st.expander("Option Parameters", expanded=True):
            s0 = st.number_input(
                "Underlying Price (S₀)",
                min_value=0.01,
                value=100.0 if not use_example else example_values["s0"],
                help="Current price of the underlying asset"
            )
            
            k = st.number_input(
                "Strike Price (K)",
                min_value=0.01,
                value=100.0 if not use_example else example_values["k"],
                help="The price at which the option holder can buy (call) or sell (put) the underlying asset"
            )
            
            t = st.number_input(
                "Time to Maturity (T in years)",
                min_value=0.01,
                max_value=30.0,
                value=1.0 if not use_example else example_values["t"],
                help="Time until the option expires (in years)"
            )
            
            r = st.number_input(
                "Risk-free Rate (r)",
                min_value=0.0,
                max_value=1.0,
                value=0.05 if not use_example else example_values["r"],
                format="%.4f",
                help="Annual risk-free interest rate (decimal form, e.g., 0.05 for 5%)"
            )
            
            sigma = st.number_input(
                "Volatility (σ)",
                min_value=0.01,
                max_value=2.0,
                value=0.2 if not use_example else example_values["sigma"],
                format="%.4f",
                help="Annual volatility of the underlying asset (decimal form)"
            )
            
            q = st.number_input(
                "Dividend Yield (q)",
                min_value=0.0,
                max_value=1.0,
                value=0.0 if not use_example else example_values["q"],
                format="%.4f",
                help="Annual dividend yield (decimal form, e.g., 0.02 for 2%)"
            )
        
        # Model selection
        pricing_model = st.selectbox(
            "Pricing Model",
            ["Black-Scholes", "Monte Carlo"],
            index=0,
            help="Select the model to price the European option"
        )
        
        if pricing_model == "Monte Carlo":
            num_sims = st.number_input(
                "Number of Simulations",
                min_value=1000,
                max_value=1000000,
                value=10000,
                step=1000,
                help="Number of Monte Carlo simulations to run"
            )
        else:
            num_sims = 10000  # Default value
    
    # Initialize the pricer
    pricer = EuropeanPricer(
        option_type=option_type.lower(),
        s0=s0,
        k=k,
        t=t,
        r=r,
        sigma=sigma,
        q=q
    )
    
    # Calculate option price and Greeks
    if pricing_model == "Black-Scholes":
        price = pricer.black_scholes_price()
        delta = pricer.delta()
        gamma = pricer.gamma()
        theta = pricer.theta()
        vega = pricer.vega()
        rho = pricer.rho()
    else:  # Monte Carlo
        price = pricer.monte_carlo_price(num_sims)
        delta, gamma, theta, vega, rho = pricer.monte_carlo_greeks(num_sims)
    
    with col2:
        st.markdown("<h3 class='subsection-header'>Results</h3>", unsafe_allow_html=True)
        
        # Display option price
        st.markdown(
            f"<div class='price-box'>"
            f"<div class='price-label'>{option_type} Option Price</div>"
            f"<div class='price-value'>${format_number(price)}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        # Display Greeks in an expandable section
        with st.expander("Option Greeks", expanded=True):
            greeks_col1, greeks_col2 = st.columns(2)
            
            with greeks_col1:
                st.markdown(
                    f"<div class='greek-box'>"
                    f"<div class='greek-label'>Delta (Δ)</div>"
                    f"<div class='greek-value'>{format_number(delta)}</div>"
                    f"<div class='greek-desc'>Rate of change of option price with respect to underlying price</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
                st.markdown(
                    f"<div class='greek-box'>"
                    f"<div class='greek-label'>Gamma (Γ)</div>"
                    f"<div class='greek-value'>{format_number(gamma)}</div>"
                    f"<div class='greek-desc'>Rate of change of Delta with respect to underlying price</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
                st.markdown(
                    f"<div class='greek-box'>"
                    f"<div class='greek-label'>Vega (ν)</div>"
                    f"<div class='greek-value'>{format_number(vega)}</div>"
                    f"<div class='greek-desc'>Rate of change of option price with respect to volatility</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            
            with greeks_col2:
                st.markdown(
                    f"<div class='greek-box'>"
                    f"<div class='greek-label'>Theta (Θ)</div>"
                    f"<div class='greek-value'>{format_number(theta)}</div>"
                    f"<div class='greek-desc'>Rate of change of option price with respect to time</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
                st.markdown(
                    f"<div class='greek-box'>"
                    f"<div class='greek-label'>Rho (ρ)</div>"
                    f"<div class='greek-value'>{format_number(rho)}</div>"
                    f"<div class='greek-desc'>Rate of change of option price with respect to interest rate</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
        
        # Display option price chart
        st.markdown("<h4 class='chart-header'>Option Price Sensitivity</h4>", unsafe_allow_html=True)
        chart_type = st.selectbox(
            "Sensitivity Analysis",
            ["Price vs. Strike", "Price vs. Volatility", "Price vs. Time to Maturity"],
            index=0
        )
        
        if chart_type == "Price vs. Strike":
            pricer.plot_price_vs_strike()
        elif chart_type == "Price vs. Volatility":
            pricer.plot_price_vs_volatility()
        else:
            pricer.plot_price_vs_time()
        
        # Download results
        st.markdown("<h4 class='download-header'>Download Results</h4>", unsafe_allow_html=True)
        if st.button("Download as CSV"):
            pricer.download_results()
            st.success("Download successful!")

elif selected_tab == "American Options":
    st.markdown("<h2 class='section-header'>American Option Pricing</h2>", unsafe_allow_html=True)
    
    # Create two columns for inputs and results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<h3 class='subsection-header'>Input Parameters</h3>", unsafe_allow_html=True)
        
        # Option example presets
        use_example = st.checkbox("Use example values", value=False)
        if use_example:
            example_values = create_example_inputs("american")
            
        # Option type selection
        option_type = st.selectbox(
            "Option Type",
            ["Call", "Put"],
            index=0 if not use_example else 0 if example_values["option_type"] == "Call" else 1
        )
        
        # Option parameters
        with st.expander("Option Parameters", expanded=True):
            s0 = st.number_input(
                "Underlying Price (S₀)",
                min_value=0.01,
                value=100.0 if not use_example else example_values["s0"],
                help="Current price of the underlying asset"
            )
            
            k = st.number_input(
                "Strike Price (K)",
                min_value=0.01,
                value=100.0 if not use_example else example_values["k"],
                help="The price at which the option holder can buy (call) or sell (put) the underlying asset"
            )
            
            t = st.number_input(
                "Time to Maturity (T in years)",
                min_value=0.01,
                max_value=30.0,
                value=1.0 if not use_example else example_values["t"],
                help="Time until the option expires (in years)"
            )
            
            r = st.number_input(
                "Risk-free Rate (r)",
                min_value=0.0,
                max_value=1.0,
                value=0.05 if not use_example else example_values["r"],
                format="%.4f",
                help="Annual risk-free interest rate (decimal form, e.g., 0.05 for 5%)"
            )
            
            sigma = st.number_input(
                "Volatility (σ)",
                min_value=0.01,
                max_value=2.0,
                value=0.2 if not use_example else example_values["sigma"],
                format="%.4f",
                help="Annual volatility of the underlying asset (decimal form)"
            )
            
            q = st.number_input(
                "Dividend Yield (q)",
                min_value=0.0,
                max_value=1.0,
                value=0.0 if not use_example else example_values["q"],
                format="%.4f",
                help="Annual dividend yield (decimal form, e.g., 0.02 for 2%)"
            )
        
        # Binomial tree parameters
        with st.expander("Binomial Tree Parameters", expanded=True):
            n_steps = st.slider(
                "Number of Time Steps (N)",
                min_value=10,
                max_value=1000,
                value=50,
                step=10,
                help="Number of time steps in the binomial tree. Higher values lead to more accurate results but slower computation."
            )
        
        # Model selection
        pricing_model = st.selectbox(
            "Pricing Model",
            ["Binomial Tree", "Least Squares Monte Carlo"],
            index=0,
            help="Select the model to price the American option"
        )
        
        if pricing_model == "Least Squares Monte Carlo":
            num_sims = st.number_input(
                "Number of Simulations",
                min_value=1000,
                max_value=100000,
                value=10000,
                step=1000,
                help="Number of Monte Carlo simulations to run"
            )
            
            n_steps_mc = st.slider(
                "Number of Time Steps for Monte Carlo",
                min_value=10,
                max_value=100,
                value=50,
                step=5,
                help="Number of time steps for the Monte Carlo simulation"
            )
        else:
            num_sims = 10000  # Default value
            n_steps_mc = 50   # Default value
    
    # Initialize the pricer
    pricer = AmericanPricer(
        option_type=option_type.lower(),
        s0=s0,
        k=k,
        t=t,
        r=r,
        sigma=sigma,
        q=q
    )
    
    # Calculate option price
    if pricing_model == "Binomial Tree":
        price, tree_data, exercise_boundary = pricer.binomial_tree_price(n_steps)
    else:  # Least Squares Monte Carlo
        price, paths, exercise_times = pricer.least_squares_mc_price(num_sims, n_steps_mc)
    
    with col2:
        st.markdown("<h3 class='subsection-header'>Results</h3>", unsafe_allow_html=True)
        
        # Display option price
        st.markdown(
            f"<div class='price-box'>"
            f"<div class='price-label'>{option_type} Option Price</div>"
            f"<div class='price-value'>${format_number(price)}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        # European equivalent price for comparison
        european_price = EuropeanPricer(
            option_type=option_type.lower(),
            s0=s0,
            k=k,
            t=t,
            r=r,
            sigma=sigma,
            q=q
        ).black_scholes_price()
        
        st.markdown(
            f"<div class='comparison-box'>"
            f"<div class='comparison-label'>European Equivalent Price</div>"
            f"<div class='comparison-value'>${format_number(european_price)}</div>"
            f"<div class='early-exercise-value'>Early Exercise Premium: ${format_number(max(0, price - european_price))}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        # Display visualization
        st.markdown("<h4 class='visualization-header'>Visualization</h4>", unsafe_allow_html=True)
        
        if pricing_model == "Binomial Tree":
            visualize_option = st.selectbox(
                "Visualization Type",
                ["Exercise Boundary", "Binomial Tree (first 5 steps)"],
                index=0
            )
            
            if visualize_option == "Exercise Boundary":
                pricer.plot_exercise_boundary(exercise_boundary)
            else:
                pricer.plot_binomial_tree(tree_data, max_steps=5)
        else:
            pricer.plot_lsmc_results(paths, exercise_times)
        
        # Download results
        st.markdown("<h4 class='download-header'>Download Results</h4>", unsafe_allow_html=True)
        if st.button("Download as CSV"):
            pricer.download_results(pricing_model.lower().replace(" ", "_"))
            st.success("Download successful!")

else:  # About section
    display_about_section()