## [Streamlit App Link](https://optionsdashboard-teq7sph6mjnsahc3a96yrq.streamlit.app/)
- Use dark mode and wide mode
- The app is not designed for mobile use
- Getting the app up on first use can take ~20 seconds

## Project Overview
The purpose of this project was to develop a dashboard that displays profit and loss, option Greeks, and volatility for call and put stock options. It can be used by a trader as part of their workflow, or by an investor looking to hedge risk in an existing position.

## Dashboard
The dashboard is broken down into two parts:

  1. **Strategy Analysis**
     - [Single](https://github.com/henrycosentino/options_dashboard/blob/main/Single.py): A single option strategy where the profit and loss for a long or short call or put can be analyzed
     - [Straddle](https://github.com/henrycosentino/options_dashboard/blob/main/pages/Straddle.py): A straddle option strategy where the profit and loss for long or short straddles can be analyzed
     - [Butterfly](https://github.com/henrycosentino/options_dashboard/blob/main/pages/Butterfly.py): Various butterfly strategies can be analyzed

  2. **Volatility Analysis**
     - [Term Structure](https://github.com/henrycosentino/options_dashboard/blob/main/pages/Volatility_Term_Structure.py): Analyzes the term structure of spot and forward volatility
     - [Surface](https://github.com/henrycosentino/options_dashboard/blob/main/pages/Volatility_Surface.py): Analyzes the current volatility surface across all traded options for the underlying

## [helpers](https://github.com/henrycosentino/options_dashboard/blob/main/helpers.py)

- **Black-Scholes & Binomial Classes**
  - Created to automate option pricing and risk metric calculations
  - Calculate the value of a call and put option, along with first and second-order Greeks
  - The pricing model can be toggled dynamically via switches in the dashboard:
    - When *American* is selected, only the Binomial class is used for pricing and Greeks
    - When *European* is selected, only the Black-Scholes class is used for pricing and Greeks
  - Black-Scholes follows the methodology from *Option Volatility and Pricing: Advanced Trading Strategies and Techniques, 2nd Edition*
  - Binomial follows the Cox-Ross-Rubinstein methodology for pricing; some Greeks are calculated by perturbing the model's price output

- **Underlying & Volatility Classes**
  - The Underlying class creates a stock object used during volatility analysis
  - The Volatility class calculates the volatility term structure (spot and forward) and the volatility surface
  - Currently, the Volatility class uses only live data; a method for calculating average spot volatility surfaces over a specified period is planned for a future update