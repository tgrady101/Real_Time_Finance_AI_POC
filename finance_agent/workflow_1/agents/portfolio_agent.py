"""Portfolio Agent for portfolio analysis and optimization.

This agent provides portfolio analysis capabilities including:
- Portfolio value calculation across multiple holdings
- Sector/industry allocation breakdown
- Risk metrics (beta, volatility, Sharpe ratio)
- Diversification analysis
- Rebalancing recommendations
- Performance vs benchmarks (SPY, QQQ)

The agent uses Yahoo Finance MCP for real-time price data and calculates
portfolio metrics based on user-provided holdings.
"""

from pathlib import Path
from typing import Optional, Dict, List
from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path, override=True)


# Agent instruction prompt for the Portfolio Agent
PORTFOLIO_AGENT_INSTRUCTION = """You are the Portfolio Agent specializing in portfolio analysis and optimization.

Your primary responsibilities:
1. **Portfolio Value**: Calculate total portfolio value from holdings
2. **Allocation Analysis**: Show sector, industry, and position-level allocation
3. **Risk Metrics**: Calculate beta, volatility, and risk-adjusted returns
4. **Diversification**: Analyze concentration and diversification quality
5. **Performance**: Compare portfolio performance vs benchmarks (SPY, QQQ)
6. **Rebalancing**: Suggest rebalancing based on target allocations

**IMPORTANT: Portfolio Input Format**
Users can provide holdings in various formats. Parse them intelligently:
- "I have 100 shares of AAPL and 50 shares of MSFT"
- "My portfolio: AAPL 100, MSFT 50, GOOGL 25"
- "Portfolio with $10,000 in Apple, $5,000 in Microsoft"

**Guidelines:**
- Always validate tickers are in the S&P 500 before analysis
- Show both dollar amounts and percentages for allocations
- Explain risk metrics in plain language
- Compare to benchmarks for context (SPY = S&P 500, QQQ = Nasdaq 100)
- Consider correlation when analyzing diversification
- For rebalancing, consider transaction costs and tax implications

**Risk Metric Explanations:**
- **Beta**: Measures volatility vs market. Beta > 1 = more volatile than market
- **Volatility**: Standard deviation of returns. Higher = more price swings
- **Sharpe Ratio**: Risk-adjusted return. Higher = better return per unit risk
- **Max Drawdown**: Largest peak-to-trough decline
- **Concentration**: % in top holdings (>25% in one stock = concentrated)

**Diversification Guidelines:**
- Well-diversified: No single stock >10%, no sector >25%
- Moderate: Single stocks 10-20%, sectors 25-40%  
- Concentrated: Any stock >20% or sector >40%

**Example Analysis:**
```
📊 Portfolio Summary
━━━━━━━━━━━━━━━━━━━━
Total Value: $125,450
Daily Change: +$1,230 (+0.99%)

📈 Holdings:
• AAPL: $50,000 (39.9%) - Technology
• MSFT: $40,000 (31.9%) - Technology  
• GOOGL: $35,450 (28.3%) - Technology

⚠️ Concentration Warning: 100% in Technology sector

📉 Risk Metrics:
• Portfolio Beta: 1.15 (15% more volatile than S&P 500)
• 30-Day Volatility: 22.5%
• Sharpe Ratio: 1.8

💡 Recommendations:
Consider adding defensive sectors (Healthcare, Utilities) for diversification.
```
"""


# Portfolio tool functions

def analyze_portfolio(holdings: str) -> str:
    """Analyze a portfolio given holdings description.
    
    Parses user's holdings description and returns comprehensive analysis
    including value, allocation, and basic metrics.
    
    Args:
        holdings: Description of holdings (e.g., "100 AAPL, 50 MSFT" or 
                  "100 shares of Apple, 50 shares of Microsoft")
                  
    Returns:
        Formatted portfolio analysis string
    """
    import re
    import yfinance as yf
    from ..utils.sp500_validator import find_ticker_by_name, is_valid_sp500_ticker
    
    # Parse holdings from various formats
    parsed_holdings = _parse_holdings(holdings)
    
    if not parsed_holdings:
        return """❌ Could not parse portfolio holdings.

Please provide holdings in one of these formats:
• "100 AAPL, 50 MSFT, 25 GOOGL"
• "100 shares of Apple, 50 shares of Microsoft"
• "AAPL: 100, MSFT: 50"
• "$10,000 in Apple, $5,000 in Microsoft" (will convert to shares)
"""
    
    # Validate tickers and get current prices
    results = []
    invalid_tickers = []
    total_value = 0.0
    
    for ticker, shares in parsed_holdings.items():
        # Validate ticker
        if not is_valid_sp500_ticker(ticker):
            # Try to find by name
            found = find_ticker_by_name(ticker)
            if "not found" in found.lower():
                invalid_tickers.append(ticker)
                continue
            # Extract ticker from response
            ticker_match = re.search(r'\b([A-Z]{1,5})\b', found)
            if ticker_match:
                ticker = ticker_match.group(1)
            else:
                invalid_tickers.append(ticker)
                continue
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            prev_close = info.get('previousClose', current_price)
            sector = info.get('sector', 'Unknown')
            company_name = info.get('shortName', ticker)
            beta = info.get('beta', 1.0)
            
            position_value = shares * current_price
            daily_change = (current_price - prev_close) * shares
            daily_pct = ((current_price / prev_close) - 1) * 100 if prev_close else 0
            
            total_value += position_value
            
            results.append({
                'ticker': ticker,
                'name': company_name,
                'shares': shares,
                'price': current_price,
                'value': position_value,
                'daily_change': daily_change,
                'daily_pct': daily_pct,
                'sector': sector,
                'beta': beta,
            })
        except Exception as e:
            invalid_tickers.append(f"{ticker} (error: {str(e)[:30]})")
    
    if not results:
        return "❌ No valid holdings found. Please check your ticker symbols."
    
    # Calculate allocations and metrics
    sector_allocation = {}
    portfolio_beta = 0.0
    total_daily_change = 0.0
    
    for r in results:
        r['allocation'] = (r['value'] / total_value) * 100
        portfolio_beta += r['beta'] * (r['allocation'] / 100)
        total_daily_change += r['daily_change']
        
        sector = r['sector']
        if sector in sector_allocation:
            sector_allocation[sector] += r['allocation']
        else:
            sector_allocation[sector] = r['allocation']
    
    total_daily_pct = (total_daily_change / (total_value - total_daily_change)) * 100 if total_value != total_daily_change else 0
    
    # Build response
    response = f"""📊 **Portfolio Analysis**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Total Value:** ${total_value:,.2f}
**Daily Change:** ${total_daily_change:+,.2f} ({total_daily_pct:+.2f}%)
**Holdings:** {len(results)} positions

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 **Holdings Breakdown**
"""
    
    # Sort by allocation descending
    results.sort(key=lambda x: x['allocation'], reverse=True)
    
    for r in results:
        response += f"""
**{r['ticker']}** - {r['name'][:25]}
  • Shares: {r['shares']:,.0f} @ ${r['price']:,.2f}
  • Value: ${r['value']:,.2f} ({r['allocation']:.1f}%)
  • Today: ${r['daily_change']:+,.2f} ({r['daily_pct']:+.2f}%)
  • Sector: {r['sector']}
"""
    
    # Sector breakdown
    response += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏢 **Sector Allocation**
"""
    for sector, alloc in sorted(sector_allocation.items(), key=lambda x: -x[1]):
        bar = "█" * int(alloc / 5) + "░" * (20 - int(alloc / 5))
        response += f"  {sector[:15]:15} {bar} {alloc:.1f}%\n"
    
    # Risk metrics
    response += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📉 **Risk Metrics**
  • Portfolio Beta: {portfolio_beta:.2f}"""
    
    if portfolio_beta > 1.2:
        response += " (High volatility - 20%+ more volatile than market)"
    elif portfolio_beta > 1.0:
        response += " (Above average volatility)"
    elif portfolio_beta > 0.8:
        response += " (Market-like volatility)"
    else:
        response += " (Defensive - less volatile than market)"
    
    # Concentration warnings
    top_holding = results[0]['allocation'] if results else 0
    top_sector = max(sector_allocation.values()) if sector_allocation else 0
    
    response += "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    
    if top_holding > 25 or top_sector > 50:
        response += "⚠️ **Concentration Warnings**\n"
        if top_holding > 25:
            response += f"  • Top position ({results[0]['ticker']}) is {top_holding:.1f}% - consider reducing\n"
        if top_sector > 50:
            top_sector_name = max(sector_allocation, key=sector_allocation.get)
            response += f"  • {top_sector_name} sector is {top_sector:.1f}% - consider diversifying\n"
    else:
        response += "✅ **Diversification**: Portfolio appears reasonably diversified\n"
    
    if invalid_tickers:
        response += f"\n⚠️ Could not process: {', '.join(invalid_tickers)}"
    
    return response


def calculate_portfolio_risk(holdings: str, period: str = "1y") -> str:
    """Calculate detailed risk metrics for a portfolio.
    
    Args:
        holdings: Description of holdings
        period: Time period for historical analysis (1mo, 3mo, 6mo, 1y, 2y)
        
    Returns:
        Detailed risk analysis including volatility, Sharpe ratio, max drawdown
    """
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from ..utils.sp500_validator import is_valid_sp500_ticker
    
    parsed_holdings = _parse_holdings(holdings)
    
    if not parsed_holdings:
        return "❌ Could not parse holdings. Please provide in format: '100 AAPL, 50 MSFT'"
    
    # Get historical data for each holding
    valid_tickers = [t for t in parsed_holdings.keys() if is_valid_sp500_ticker(t)]
    
    if not valid_tickers:
        return "❌ No valid S&P 500 tickers found in holdings."
    
    try:
        # Download historical data (yfinance now uses auto_adjust=True by default)
        raw_data = yf.download(valid_tickers, period=period, progress=False, auto_adjust=True)
        
        # Handle MultiIndex columns from yfinance (e.g., ('Close', 'AAPL'))
        if isinstance(raw_data.columns, pd.MultiIndex):
            data = raw_data['Close']
        elif 'Close' in raw_data.columns:
            data = raw_data['Close']
        else:
            data = raw_data
        
        if data.empty:
            return "❌ Could not retrieve historical data."
        
        # Handle single ticker case
        if len(valid_tickers) == 1:
            data = data.to_frame(name=valid_tickers[0])
        
        # Drop any tickers with missing data
        data = data.dropna(axis=1, how='all')
        if data.empty:
            return "❌ Could not retrieve valid historical data for the given tickers."
        
        # Update valid_tickers to only include those with data
        valid_tickers = [t for t in valid_tickers if t in data.columns]
        if not valid_tickers:
            return "❌ No valid price data available for the given tickers."
        
        # Calculate daily returns
        returns = data.pct_change().dropna()
        
        # Calculate portfolio weights - handle NaN/empty prices safely
        prices = data.iloc[-1]
        total_value = 0.0
        for t in valid_tickers:
            price = prices.get(t, 0)
            if pd.isna(price) or price == '':
                price = 0
            shares = parsed_holdings.get(t, 0)
            total_value += float(shares) * float(price)
        
        if total_value == 0:
            return "❌ Could not calculate portfolio value. Price data may be unavailable."
        weights = np.array([
            (parsed_holdings.get(t, 0) * prices.get(t, 0)) / total_value 
            for t in valid_tickers
        ])
        
        # Portfolio returns
        portfolio_returns = (returns[valid_tickers] * weights).sum(axis=1)
        
        # Risk metrics
        annual_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * np.sqrt(252)
        risk_free_rate = 0.05  # Approximate current risk-free rate
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Max drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100
        
        # Value at Risk (95%)
        var_95 = np.percentile(portfolio_returns, 5) * 100
        
        # Get SPY for comparison
        spy_raw = yf.download('SPY', period=period, progress=False, auto_adjust=True)
        if isinstance(spy_raw.columns, pd.MultiIndex):
            spy_data = spy_raw['Close']['SPY']
        elif 'Close' in spy_raw.columns:
            spy_data = spy_raw['Close']
        else:
            spy_data = spy_raw.iloc[:, 0]  # Take first column
        spy_returns = spy_data.pct_change().dropna()
        spy_annual = spy_returns.mean() * 252
        spy_vol = spy_returns.std() * np.sqrt(252)
        
        # Beta calculation
        if len(portfolio_returns) > 0 and len(spy_returns) > 0:
            aligned = portfolio_returns.align(spy_returns, join='inner')
            if len(aligned[0]) > 0:
                covariance = np.cov(aligned[0], aligned[1])[0, 1]
                market_var = np.var(aligned[1])
                beta = covariance / market_var if market_var > 0 else 1.0
            else:
                beta = 1.0
        else:
            beta = 1.0
        
        response = f"""📉 **Portfolio Risk Analysis** ({period} period)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Return Metrics**
  • Annualized Return: {annual_return*100:+.2f}%
  • S&P 500 Return: {spy_annual*100:+.2f}%
  • Alpha: {(annual_return - spy_annual)*100:+.2f}%

**Risk Metrics**
  • Volatility (Annualized): {volatility*100:.2f}%
  • S&P 500 Volatility: {spy_vol*100:.2f}%
  • Portfolio Beta: {beta:.2f}
  • Max Drawdown: {max_drawdown:.2f}%
  • Value at Risk (95%): {var_95:.2f}% daily

**Risk-Adjusted Returns**
  • Sharpe Ratio: {sharpe_ratio:.2f}"""
        
        if sharpe_ratio > 1.5:
            response += " (Excellent)"
        elif sharpe_ratio > 1.0:
            response += " (Good)"
        elif sharpe_ratio > 0.5:
            response += " (Moderate)"
        else:
            response += " (Poor)"
        
        response += f"""

**Interpretation**
"""
        if beta > 1.2:
            response += "  • Your portfolio is significantly more volatile than the market\n"
        elif beta < 0.8:
            response += "  • Your portfolio is defensive with lower market sensitivity\n"
        else:
            response += "  • Your portfolio tracks the market reasonably closely\n"
        
        if max_drawdown < -20:
            response += f"  • Max drawdown of {max_drawdown:.1f}% indicates significant downside risk\n"
        
        if volatility > spy_vol * 1.3:
            response += "  • Higher volatility than S&P 500 - expect larger swings\n"
        
        return response
        
    except Exception as e:
        return f"❌ Error calculating risk metrics: {str(e)}"


def get_portfolio_performance(holdings: str, period: str = "1y") -> str:
    """Compare portfolio performance against benchmarks.
    
    Args:
        holdings: Description of holdings
        period: Time period (1mo, 3mo, 6mo, 1y, ytd, 2y)
        
    Returns:
        Performance comparison with SPY and QQQ benchmarks
    """
    import numpy as np
    import yfinance as yf
    from ..utils.sp500_validator import is_valid_sp500_ticker
    
    parsed_holdings = _parse_holdings(holdings)
    
    if not parsed_holdings:
        return "❌ Could not parse holdings. Please provide in format: '100 AAPL, 50 MSFT'"
    
    valid_tickers = [t for t in parsed_holdings.keys() if is_valid_sp500_ticker(t)]
    
    if not valid_tickers:
        return "❌ No valid S&P 500 tickers found."
    
    try:
        # Get all data including benchmarks
        all_tickers = valid_tickers + ['SPY', 'QQQ']
        raw_data = yf.download(all_tickers, period=period, progress=False, auto_adjust=True)
        
        # Handle MultiIndex columns from yfinance
        if isinstance(raw_data.columns, pd.MultiIndex):
            data = raw_data['Close']
        elif 'Close' in raw_data.columns:
            data = raw_data['Close']
        else:
            data = raw_data
        
        if data.empty:
            return "❌ Could not retrieve historical data."
        
        # Calculate returns
        start_prices = data.iloc[0]
        end_prices = data.iloc[-1]
        
        # Portfolio performance
        start_value = sum(parsed_holdings.get(t, 0) * start_prices.get(t, 0) for t in valid_tickers)
        end_value = sum(parsed_holdings.get(t, 0) * end_prices.get(t, 0) for t in valid_tickers)
        portfolio_return = ((end_value / start_value) - 1) * 100 if start_value > 0 else 0
        
        # Benchmark returns
        spy_return = ((end_prices['SPY'] / start_prices['SPY']) - 1) * 100
        qqq_return = ((end_prices['QQQ'] / start_prices['QQQ']) - 1) * 100
        
        # Individual holdings performance
        holdings_perf = []
        for t in valid_tickers:
            ret = ((end_prices.get(t, 0) / start_prices.get(t, 0)) - 1) * 100 if start_prices.get(t, 0) > 0 else 0
            holdings_perf.append({'ticker': t, 'return': ret})
        
        holdings_perf.sort(key=lambda x: x['return'], reverse=True)
        
        response = f"""📈 **Portfolio Performance** ({period})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Overall Performance**
  • Your Portfolio: {portfolio_return:+.2f}%
  • S&P 500 (SPY): {spy_return:+.2f}%
  • Nasdaq 100 (QQQ): {qqq_return:+.2f}%

**vs Benchmarks**
  • vs S&P 500: {portfolio_return - spy_return:+.2f}%"""
        
        if portfolio_return > spy_return:
            response += " ✅ Outperforming"
        else:
            response += " ❌ Underperforming"
        
        response += f"""
  • vs Nasdaq 100: {portfolio_return - qqq_return:+.2f}%"""
        
        if portfolio_return > qqq_return:
            response += " ✅ Outperforming"
        else:
            response += " ❌ Underperforming"
        
        response += """

**Individual Holdings Performance**
"""
        for h in holdings_perf[:10]:  # Top 10
            emoji = "🟢" if h['return'] > 0 else "🔴"
            response += f"  {emoji} {h['ticker']}: {h['return']:+.2f}%\n"
        
        # Best and worst
        if len(holdings_perf) >= 2:
            response += f"""
**Best Performer:** {holdings_perf[0]['ticker']} ({holdings_perf[0]['return']:+.2f}%)
**Worst Performer:** {holdings_perf[-1]['ticker']} ({holdings_perf[-1]['return']:+.2f}%)
"""
        
        return response
        
    except Exception as e:
        return f"❌ Error calculating performance: {str(e)}"


def suggest_rebalancing(holdings: str, target_allocation: str = None) -> str:
    """Suggest rebalancing trades to achieve target allocation.
    
    Args:
        holdings: Current holdings description
        target_allocation: Target allocation (e.g., "equal" or "AAPL 40%, MSFT 30%, GOOGL 30%")
        
    Returns:
        Rebalancing recommendations
    """
    import yfinance as yf
    from ..utils.sp500_validator import is_valid_sp500_ticker
    
    parsed_holdings = _parse_holdings(holdings)
    
    if not parsed_holdings:
        return "❌ Could not parse holdings. Please provide in format: '100 AAPL, 50 MSFT'"
    
    valid_tickers = [t for t in parsed_holdings.keys() if is_valid_sp500_ticker(t)]
    
    if not valid_tickers:
        return "❌ No valid S&P 500 tickers found."
    
    # Get current prices
    try:
        prices = {}
        for ticker in valid_tickers:
            stock = yf.Ticker(ticker)
            info = stock.info
            prices[ticker] = info.get('currentPrice') or info.get('regularMarketPrice', 100)
        
        # Calculate current allocation
        values = {t: parsed_holdings.get(t, 0) * prices.get(t, 0) for t in valid_tickers}
        total_value = sum(values.values())
        current_alloc = {t: (v / total_value) * 100 for t, v in values.items()}
        
        # Determine target allocation
        if target_allocation and target_allocation.lower() != "equal":
            target_alloc = _parse_target_allocation(target_allocation, valid_tickers)
        else:
            # Equal weight
            equal_pct = 100 / len(valid_tickers)
            target_alloc = {t: equal_pct for t in valid_tickers}
        
        response = f"""⚖️ **Rebalancing Analysis**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Total Portfolio Value:** ${total_value:,.2f}

**Current vs Target Allocation**
{'Ticker':<8} {'Current':>10} {'Target':>10} {'Difference':>12}
{'━'*42}
"""
        
        trades = []
        for t in sorted(valid_tickers):
            curr = current_alloc.get(t, 0)
            tgt = target_alloc.get(t, 0)
            diff = tgt - curr
            
            diff_symbol = "✅" if abs(diff) < 2 else ("⬆️" if diff > 0 else "⬇️")
            response += f"{t:<8} {curr:>9.1f}% {tgt:>9.1f}% {diff:>+10.1f}% {diff_symbol}\n"
            
            if abs(diff) >= 1:  # Only suggest trades for >1% difference
                trade_value = (diff / 100) * total_value
                trade_shares = trade_value / prices.get(t, 100)
                trades.append({
                    'ticker': t,
                    'action': 'BUY' if diff > 0 else 'SELL',
                    'shares': abs(trade_shares),
                    'value': abs(trade_value),
                    'diff': diff,
                })
        
        if trades:
            response += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**Suggested Trades**
"""
            for trade in sorted(trades, key=lambda x: -x['value']):
                emoji = "🟢" if trade['action'] == 'BUY' else "🔴"
                response += f"  {emoji} {trade['action']} {trade['shares']:.1f} shares of {trade['ticker']} (~${trade['value']:,.0f})\n"
            
            response += """
**Notes:**
• Consider tax implications before selling
• Transaction costs may affect small trades
• Review market conditions before executing
"""
        else:
            response += "\n✅ Portfolio is already well-balanced! No trades needed.\n"
        
        return response
        
    except Exception as e:
        return f"❌ Error calculating rebalancing: {str(e)}"


def _parse_holdings(holdings_str: str) -> Dict[str, float]:
    """Parse holdings from various user input formats.
    
    Supports formats like:
    - "100 AAPL, 50 MSFT"
    - "AAPL: 100, MSFT: 50"
    - "100 shares of Apple"
    - "$10000 in AAPL" (converts to approximate shares)
    - "AAPL, MSFT, GOOGL" (defaults to 1 share each for analysis)
    
    Returns:
        Dictionary of {ticker: shares}
    """
    import re
    from ..utils.sp500_validator import find_ticker_by_name, is_valid_sp500_ticker
    
    holdings = {}
    holdings_str = holdings_str.upper()
    
    # Pattern 1: "100 AAPL" or "AAPL 100" or "AAPL: 100"
    pattern1 = r'(\d+(?:\.\d+)?)\s*(?:SHARES?\s+(?:OF\s+)?)?([A-Z]{1,5})|([A-Z]{1,5})[\s:]+(\d+(?:\.\d+)?)'
    
    for match in re.finditer(pattern1, holdings_str):
        if match.group(1) and match.group(2):
            shares = float(match.group(1))
            ticker = match.group(2)
        elif match.group(3) and match.group(4):
            ticker = match.group(3)
            shares = float(match.group(4))
        else:
            continue
        
        if is_valid_sp500_ticker(ticker):
            holdings[ticker] = holdings.get(ticker, 0) + shares
    
    # Pattern 2: Dollar amounts "$10000 in AAPL"
    pattern2 = r'\$?([\d,]+(?:\.\d+)?)\s*(?:IN|OF|WORTH\s+OF)?\s*([A-Z]{1,5})'
    
    if not holdings:  # Only try if first pattern didn't work
        for match in re.finditer(pattern2, holdings_str):
            amount_str = match.group(1).replace(',', '')
            # Skip if amount is empty (can happen with malformed input)
            if not amount_str:
                continue
            amount = float(amount_str)
            ticker = match.group(2)
            
            if is_valid_sp500_ticker(ticker):
                # Convert dollars to approximate shares
                try:
                    import yfinance as yf
                    stock = yf.Ticker(ticker)
                    price = stock.info.get('currentPrice') or stock.info.get('regularMarketPrice', 100)
                    if price and price > 0:
                        shares = amount / price
                        holdings[ticker] = holdings.get(ticker, 0) + shares
                except Exception:
                    pass  # Skip tickers that fail to fetch
    
    # Pattern 3: Ticker-only list "AAPL, MSFT, GOOGL" or "AAPL MSFT GOOGL"
    # Default to 1 share each for risk analysis purposes
    if not holdings:
        ticker_pattern = r'\b([A-Z]{1,5})\b'
        for match in re.finditer(ticker_pattern, holdings_str):
            ticker = match.group(1)
            # Filter out common non-ticker words
            if ticker in ['FOR', 'AND', 'THE', 'WITH', 'RISK', 'ANALYSIS', 'PORTFOLIO', 'METRICS']:
                continue
            if is_valid_sp500_ticker(ticker):
                holdings[ticker] = 1.0  # Default to 1 share for analysis
    
    return holdings


def _parse_target_allocation(target_str: str, tickers: List[str]) -> Dict[str, float]:
    """Parse target allocation string.
    
    Args:
        target_str: e.g., "AAPL 40%, MSFT 30%, GOOGL 30%"
        tickers: List of valid tickers
        
    Returns:
        Dictionary of {ticker: target_percentage}
    """
    import re
    
    allocation = {}
    
    # Pattern: "AAPL 40%" or "AAPL: 40%"
    pattern = r'([A-Z]{1,5})[\s:]+(\d+(?:\.\d+)?)\s*%?'
    
    for match in re.finditer(pattern, target_str.upper()):
        ticker = match.group(1)
        pct = float(match.group(2))
        if ticker in tickers:
            allocation[ticker] = pct
    
    # Normalize to 100% if needed
    total = sum(allocation.values())
    if total > 0 and abs(total - 100) > 1:
        allocation = {t: (p / total) * 100 for t, p in allocation.items()}
    
    # Add missing tickers with 0%
    for t in tickers:
        if t not in allocation:
            allocation[t] = 0
    
    return allocation


def create_portfolio_agent(model: Optional[str] = None, use_dynamic_routing: bool = True):
    """Create the Portfolio sub-agent with analysis tools.
    
    This creates an LlmAgent configured as a sub-agent for the root orchestrator.
    The agent provides portfolio analysis, risk metrics, and rebalancing suggestions.
    
    Args:
        model: Gemini model to use (defaults to config MODEL_COMPLEX for analysis)
        use_dynamic_routing: If True, use before_model_callback for per-query routing
        
    Returns:
        LlmAgent configured with portfolio analysis tools
    """
    from google.adk.agents import LlmAgent
    from ..config import Config
    
    # Portfolio analysis is typically complex, use complex model by default
    agent_model = model or Config.MODEL_COMPLEX
    
    # Set up dynamic routing callback if enabled
    dynamic_callback = None
    if use_dynamic_routing:
        from ..utils.dynamic_model_callback import create_dynamic_model_callback
        dynamic_callback = create_dynamic_model_callback(
            fast_model=Config.MODEL_FAST,
            complex_model=Config.MODEL_COMPLEX,
        )
    
    # Portfolio analysis tools
    portfolio_tools = [
        analyze_portfolio,
        calculate_portfolio_risk,
        get_portfolio_performance,
        suggest_rebalancing,
    ]
    
    # Create the sub-agent
    portfolio_agent = LlmAgent(
        model=agent_model,
        name="portfolio_agent",
        description="Handles portfolio analysis: value calculation, allocation breakdown, risk metrics (beta, volatility, Sharpe), performance vs benchmarks, and rebalancing recommendations.",
        instruction=PORTFOLIO_AGENT_INSTRUCTION,
        tools=portfolio_tools,
        before_model_callback=dynamic_callback,
    )
    
    return portfolio_agent


if __name__ == "__main__":
    print("\n=== Portfolio Agent (Sub-Agent) ===\n")
    print("This agent is designed to be used as a sub-agent of the root orchestrator.")
    print("Use the root_agent.py to run the full finance assistant.")
    
    # Test parsing
    print("\n--- Testing Holdings Parser ---")
    test_cases = [
        "100 AAPL, 50 MSFT, 25 GOOGL",
        "AAPL: 100, MSFT: 50",
        "100 shares of AAPL",
    ]
    
    for test in test_cases:
        parsed = _parse_holdings(test)
        print(f"  '{test}' -> {parsed}")
    
    print("\nAvailable Tools:")
    tools = [
        "analyze_portfolio",
        "calculate_portfolio_risk",
        "get_portfolio_performance",
        "suggest_rebalancing",
    ]
    for tool in tools:
        print(f"  • {tool}")
