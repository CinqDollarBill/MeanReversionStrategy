import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('LCRXAMAT.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace = True)
    #meanReversion(df)

    # Run backtest 
    trades_df, portfolio_series, pnl = backtest_mean_reversion(df, 1.54, -1.54, 14)
    
    print("----- Strategy Performance Metrics from pnl array -----")
    print(f"Total Return    : ${pnl[2]:,.2f}")
    print(f"Upper Z         : {pnl[5]:.2f}")
    print(f"Lower Z         : {pnl[6]:.2f}")
    print(f"Base Risk       : {pnl[7]:.2%}")
    print(f"Window          : {pnl[8]}")
    print(f"Number of Trades: {pnl[3]}")
    print(f"Win Rate        : {pnl[4]:.2%}")
    print(f"Sharpe Ratio    : {pnl[9]:.4f}")
    print("---------------------------------------------------------")
    
    # Calculate Maximum Drawdown
    cumulative_max = portfolio_series.cummax()
    drawdown = (portfolio_series - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    print(f"Maximum Drawdown: {max_drawdown:.2%}")

    # Show first 5 trades
    print("\nFirst 5 trades:")
    print(trades_df.head())
    print(trades_df.tail())

    # Calculate Sharpe Ratio
    returns = portfolio_series.pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    
    # Calculate Maximum Drawdown
    cumulative_max = portfolio_series.cummax()
    drawdown = (portfolio_series - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    print(f"Maximum Drawdown: {max_drawdown:.2%}")

    # Plot portfolio value
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,6))
    plt.plot(portfolio_series)
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Value ($)")
    plt.grid(True)
    plt.show()

    meanReversion(df, 1.54, -1.54, 14)
    

def backtest_mean_reversion(df, upperZ, lowerZ, window):
    # Calculate spread and z-score over a rolling window
    # window = 30
    df['Spread'] = df['LCRX'] - df['AMAT']
    df['Rolling_Mean'] = df['Spread'].rolling(window).mean()
    df['Rolling_Std'] = df['Spread'].rolling(window).std()
    df['Z-Score'] = (df['Spread'] - df['Rolling_Mean']) / df['Rolling_Std']
    
    # Compute rolling volume averages for each stock
    df['LCRX_Vol_MA'] = df['LCRX Volume'].rolling(window).mean()
    df['AMAT_Vol_MA'] = df['AMAT Volume'].rolling(window).mean()
    
    # Strategy parameters
    initial_capital = 100_000
    # base_risk = 0.05  # base risk per trade: 5% of capital
    portfolio_value = [initial_capital]
    positions = []
    
    # Trade tracking variables
    current_position = None
    entry_price_lcrx = None
    entry_price_amat = None
    entry_date = None

        # new based fully on volume
    for i in range(len(df)):
        if pd.isna(df.iloc[i]['Z-Score']):
            continue

        current_capital = portfolio_value[-1]
        z = df.iloc[i]['Z-Score']
        lcrx_price = df.iloc[i]['LCRX']
        amat_price = df.iloc[i]['AMAT']
        date = df.index[i]

        # Get current trading volumes and moving averages
        current_lcrx_vol = df.iloc[i]['LCRX Volume']
        current_amat_vol = df.iloc[i]['AMAT Volume']
        lcrx_vol_ma = df.iloc[i]['LCRX_Vol_MA']
        amat_vol_ma = df.iloc[i]['AMAT_Vol_MA']

        # Volume-based position sizing
        lcrx_trade_size = current_lcrx_vol / lcrx_vol_ma if lcrx_vol_ma else 1
        amat_trade_size = current_amat_vol / amat_vol_ma if amat_vol_ma else 1

        # Normalize trade sizes to avoid extreme values
        trade_size = (lcrx_trade_size + amat_trade_size) / 2

        if current_position:
            if (current_position == 'long' and z >= 0) or (current_position == 'short' and z <= 0):
                lcrx_return = (lcrx_price - entry_price_lcrx) / entry_price_lcrx
                amat_return = (entry_price_amat - amat_price) / entry_price_amat
                pnl = current_capital * trade_size * (lcrx_return + amat_return)

                portfolio_value.append(portfolio_value[-1] + pnl)
                positions.append({
                    'Entry': entry_date,
                    'Exit': date,
                    'Type': current_position,
                    'Trade Size': trade_size,
                    'LCRX Entry': entry_price_lcrx,
                    'LCRX Curr Price': lcrx_price,
                    'AMAT Entry': entry_price_amat,
                    'AMAT Curr Price': amat_price,
                    'PnL': pnl
                })
                current_position = None
            else:
                portfolio_value.append(portfolio_value[-1])
        else:
            if z < lowerZ:  # Long LCRX, Short AMAT
                current_position = 'long'
                entry_price_lcrx = lcrx_price
                entry_price_amat = amat_price
                entry_date = date
            elif z > upperZ:  # Short LCRX, Long AMAT
                current_position = 'short'
                entry_price_lcrx = lcrx_price
                entry_price_amat = amat_price
                entry_date = date

            portfolio_value.append(portfolio_value[-1])

    # Final performance metrics
    total_return = portfolio_value[-1] - initial_capital
    num_trades = len(positions)
    win_rate = len([p for p in positions if p['PnL'] > 0]) / num_trades if num_trades > 0 else 0

    returns = pd.Series(portfolio_value, index=df.index[:len(portfolio_value)]).pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
    
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Value: ${portfolio_value[-1]:,.2f}")
    print(f"Total Return: ${total_return:,.2f} ({total_return/initial_capital:.1%})")
    print(f"Number of Trades: {num_trades}")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")

    pnl = []
    pnl.extend([initial_capital, portfolio_value[-1], total_return, num_trades, win_rate, upperZ, lowerZ, trade_size, window, sharpe_ratio])
    
    return pd.DataFrame(positions), pd.Series(portfolio_value, index=df.index[:len(portfolio_value)]), pnl

def meanReversion(df, upperZ, lowerZ, window):
    # Calculate spread (1:1 ratio for simplicity)
    df['Spread'] = df['LCRX'] - df['AMAT']

    # Compute rolling mean and std (30-day window)
    df['Rolling_Mean'] = df['Spread'].rolling(window).mean()
    df['Rolling_Std'] = df['Spread'].rolling(window).std()
    df['Z-Score'] = (df['Spread'] - df['Rolling_Mean']) / df['Rolling_Std']

    # Plot Z-Score
    plt.figure(figsize=(12,6))
    plt.plot(df['Z-Score'], label='Z-Score')
    plt.axhline(2, color='r', linestyle='--', label='Overvalued Threshold')
    plt.axhline(-2, color='g', linestyle='--', label='Undervalued Threshold')

    # Generate signals
    df['Long_Signal'] = np.where(df['Z-Score'] < lowerZ, 1, 0)  # Buy LCRX, Sell AMAT
    df['Short_Signal'] = np.where(df['Z-Score'] > upperZ, -1, 0)  # Sell LCRX, Buy AMAT

    # Plot signals on the graph
    plt.scatter(df.index[df['Long_Signal'] == 1], df['Z-Score'][df['Long_Signal'] == 1], 
                color='green', marker='^', label='Long LCRX/Short AMAT', s=100)
    plt.scatter(df.index[df['Short_Signal'] == -1], df['Z-Score'][df['Short_Signal'] == -1], 
                color='red', marker='v', label='Short LCRX/Long AMAT', s=100)

    plt.legend()
    plt.title('Z-Score of LCRX-AMAT Spread')
    plt.show()

if __name__ == "__main__":
    main()
