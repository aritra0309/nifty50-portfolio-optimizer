
import numpy as np
import pandas as pd

def calculate_portfolio_metrics(weights, returns, cov_matrix, risk_free_rate=0.06):
    """
    Calculate portfolio return, volatility, and Sharpe ratio
    """
    portfolio_return = np.sum(returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio

def optimize_custom_portfolio(portfolio_data, enhanced_stock_data, num_stocks=None, 
                               target_return=None, min_sharpe=None, max_risk=None, 
                               risk_free_rate=0.06):
    """
    Create custom portfolio based on user preferences
    
    Parameters:
    - portfolio_data: Dictionary with stock metrics
    - enhanced_stock_data: Historical stock data
    - num_stocks: Number of stocks to include (None = use all available)
    - target_return: Minimum expected return (e.g., 0.15 for 15%)
    - min_sharpe: Minimum Sharpe ratio required
    - max_risk: Maximum volatility/risk allowed (e.g., 0.25 for 25%)
    - risk_free_rate: Risk-free rate for Sharpe calculation
    
    Returns:
    - Dictionary with portfolio details
    """
    
    # Get available stocks
    available_tickers = list(portfolio_data.keys())
    
    # Filter by number of stocks if specified
    if num_stocks and num_stocks < len(available_tickers):
        # Sort by expected return and take top N
        sorted_tickers = sorted(available_tickers, 
                               key=lambda x: portfolio_data[x]['expected_return'], 
                               reverse=True)
        selected_tickers = sorted_tickers[:num_stocks]
    else:
        selected_tickers = available_tickers
    
    # Extract returns and covariance for selected stocks
    returns_array = np.array([portfolio_data[t]['expected_return'] for t in selected_tickers])
    
    # Calculate covariance from historical returns
    historical_returns_matrix = pd.DataFrame({
        ticker: enhanced_stock_data[ticker]['returns'].values 
        for ticker in selected_tickers
    })
    cov_matrix = historical_returns_matrix.cov().values
    
    # Generate random portfolios
    num_portfolios = 50000
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    np.random.seed(42)
    valid_portfolios = []
    
    for i in range(num_portfolios):
        weights = np.random.random(len(selected_tickers))
        weights /= np.sum(weights)
        
        p_return, p_volatility, p_sharpe = calculate_portfolio_metrics(
            weights, returns_array, cov_matrix, risk_free_rate
        )
        
        # Check constraints
        valid = True
        if target_return and p_return < target_return:
            valid = False
        if min_sharpe and p_sharpe < min_sharpe:
            valid = False
        if max_risk and p_volatility > max_risk:
            valid = False
        
        if valid:
            valid_portfolios.append(i)
            weights_record.append(weights)
            results[0, i] = p_return
            results[1, i] = p_volatility
            results[2, i] = p_sharpe
        else:
            results[0, i] = np.nan
            results[1, i] = np.nan
            results[2, i] = np.nan
    
    if not valid_portfolios:
        return None
    
    # Find best portfolio among valid ones
    valid_sharpe = results[2, valid_portfolios]
    best_idx_in_valid = np.argmax(valid_sharpe)
    best_idx = valid_portfolios[best_idx_in_valid]
    
    optimal_weights = weights_record[best_idx_in_valid]
    optimal_return = results[0, best_idx]
    optimal_volatility = results[1, best_idx]
    optimal_sharpe = results[2, best_idx]
    
    # Create results dictionary
    portfolio_result = {
        'tickers': selected_tickers,
        'weights': optimal_weights,
        'expected_return': optimal_return,
        'volatility': optimal_volatility,
        'sharpe_ratio': optimal_sharpe,
        'num_valid_portfolios': len(valid_portfolios),
        'total_portfolios_tested': num_portfolios,
        'portfolio_data': portfolio_data
    }
    
    return portfolio_result
