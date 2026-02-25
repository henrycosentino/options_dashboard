## [Streamlit App Link](https://optionsdashboard-teq7sph6mjnsahc3a96yrq.streamlit.app/)
- Use dark mode and wide mode, and the app is not designed for mobile use 
- Getting the app 'up' on first use can take ~20 seconds

## Project Overview
- The purpose of this project was to develop a dashboard that displays profit and loss, option Greeks, and volatility for a call or put stock option. A trader could use the dashboard in their workflow, or an investor looking to hedge risk in one of their positions.

## Dashboard
- The dashboard can be broken down into two parts: Strategy Analysis & Volatility Analysis

  - Strategy Analysis
    - [Single](https://github.com/henrycosentino/options_dashboard/blob/main/Single.py): a single option strategy, where the profit and loss for a long or short, call or put option can be analyzed
    - [Straddle](https://github.com/henrycosentino/options_dashboard/blob/main/pages/Straddle.py): a straddle option strategy, where the profit and loss for long or short straddles can be analyzed
    - [Butterfly](https://github.com/henrycosentino/options_dashboard/blob/main/pages/Butterfly.py): various butterfly strategies can be analyzed

  - Volatility Analysis
    - [Term Structure](https://github.com/henrycosentino/options_dashboard/blob/main/pages/Volatility_Term_Structure.py): analyzes the term structure of spot and forward volatility
    - [Surface](https://github.com/henrycosentino/options_dashboard/blob/main/pages/Volatility_Surface.py): analyzes the current volatility surface of all traded options for the underlying
            
## [helpers](https://github.com/henrycosentino/options_dashboard/blob/main/helpers.py)
- Black-Scholes & Binomial Classes
  - The classes were created to automate the process of various option pricing and risk metric calculations
  - They calculate the value of a call and put option, along with first and second-order option Greeks
  - Black-Scholes and Binomial can be dynamically set via switches inside the dashboard
      - When 'American' is selected, only the Binomial class is used for pricing and Greeks
      - When 'European' is selected, only the BlackScholes class is used for pricing and Greeks
  - Black-Scholes relies on the methodology expressed in Option Volatility and Pricing: Advanced Trading Strategies and Techniques, 2nd Edition
  - Binomial relies on the methodology expressed by Cox-Ross-Rubinstein to price call and put options, while some of the Greeks are calculated by perturbing the option price output of the Cox-Ross-Rubinstein model

- Underlying & Volatility Classes
  - The Underlying class is used to create a stock object used during the process of volatility analysis
  - The Volatility class is used to calculate the volatility term structure (spot and forward) and the volatility surface
  - Currently, the Volatility class only uses live data. However, in the future, a method for determining the average spot volatility surfaces over a specific period will be implemented