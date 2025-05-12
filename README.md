# Option Pricer App

A comprehensive Streamlit web application for pricing European and American options using quantitative finance models.

## Features

### European Options
- Black-Scholes analytical model for pricing
- Monte Carlo simulation alternative
- Greeks calculation (Delta, Gamma, Vega, Theta, Rho)
- Visualization of option price sensitivities

### American Options
- Binomial Tree (Cox-Ross-Rubinstein) model
- Least Squares Monte Carlo simulation
- Early exercise boundary visualization
- Binomial tree visualization

## Project Structure

```
option_pricer_app/
├── app.py                 # Main Streamlit application
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── european_pricer.py # European option pricing functionality
│   ├── american_pricer.py # American option pricing functionality
│   └── helpers.py         # Helper functions for formatting and examples
├── models/                # Pricing models implementation
│   ├── __init__.py
│   ├── binomial_tree.py   # Binomial tree model for American options
│   └── black_scholes.py   # Black-Scholes model for European options
├── assets/                # Static assets
│   └── style.css          # Custom CSS styling
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/option-pricer-app.git
cd option-pricer-app
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the application:
```
streamlit run app.py
```

## Usage

### European Option Pricing
1. Select "European Options" in the sidebar
2. Choose between Call or Put option
3. Enter option parameters (underlying price, strike, time to maturity, etc.)
4. View the option price and Greeks
5. Analyze the price sensitivity charts
6. Download results as CSV if needed

### American Option Pricing
1. Select "American Options" in the sidebar
2. Choose between Call or Put option
3. Enter option parameters
4. Adjust the number of time steps for the binomial tree
5. View the option price and early exercise premium
6. Explore the visualization of the exercise boundary or binomial tree
7. Download results as CSV if needed

## Mathematical Background

### Black-Scholes Formula
For a European call option:
```
C = S₀e⁻ᵈᵗN(d₁) - Ke⁻ʳᵗN(d₂)
```

For a European put option:
```
P = Ke⁻ʳᵗN(-d₂) - S₀e⁻ᵈᵗN(-d₁)
```

where:
```
d₁ = [ln(S₀/K) + (r - q + σ²/2)t] / (σ√t)
d₂ = d₁ - σ√t
```

### Binomial Tree Model
The Cox-Ross-Rubinstein binomial model uses up and down factors:
```
u = e^(σ√Δt)
d = 1/u
```

With risk-neutral probability:
```
p = (e^((r-q)Δt) - d) / (u - d)
```

## References
- Hull, J. C. (2018). *Options, Futures, and Other Derivatives*. Pearson.
- Wilmott, P. (2006). *Paul Wilmott On Quantitative Finance*. Wiley.
- Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options by simulation: A simple least-squares approach. *The Review of Financial Studies*, 14(1), 113-147.

## Disclaimer
This application is provided for educational and informational purposes only. It is not intended as financial advice or as a recommendation to trade options. Options trading involves significant risk and is not suitable for all investors.

## License
MIT License