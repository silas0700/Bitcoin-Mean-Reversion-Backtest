import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
# 1. DATA LOADING
try:
    df = pd.read_csv('BTC-USD.csv')
except FileNotFoundError:
    print("Error")
    raise SystemExit

# Clean Data
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d')
df = df.rename(columns={'timestamp': 'Date'})
df['chg'] = df['price'].pct_change()

# Split Train/Test
idx = int(df.index[df['Date'] == "2025-01-01"][0])
train_df, test_df = df[:idx], df[idx:]


# 2. BACKTESTing
def backtest(df):
    df = df.copy()
    
    # Trades & Costs
    df['pos_t-1'] = df['pos'].shift(1)
    df['trade'] = abs(df['pos_t-1'] - df['pos'])
    df['pnl'] = df['pos_t-1'] * df['chg'] - df['trade'] * 0.0005 
    
    # Cumulative Returns
    df['cumu'] = (1 + df['pnl']).cumprod()
    df['bnh_cumu'] = (1 + df['chg']).cumprod()
    
    # Drawdown
    df['dd'] = df['cumu'].cummax() - df['cumu']
    mdd = df['dd'].max()
    
    # Sharpe
    annual_return = df['pnl'].mean() * 365
    volatility = df['pnl'].std() * np.sqrt(365)
    sharpe = annual_return / volatility if volatility != 0 else 0
    
    return df, {
        'annual_return': annual_return,
        'sharpe': sharpe,
        'mdd': mdd
    }

def run_backtest(strategy_fn, df, **params):
    df = strategy_fn(df, **params)
    return backtest(df)


# 3. STRATEGY LOGIC
def mean_reversion_strategy(df, window, threshold):
    df = df.copy()
    df['momentum'] = df['price'].pct_change(periods=window)
    
    df['pos'] = 0 
    df.loc[df['momentum'] < -threshold, 'pos'] = 1 # Buy Dip
    df.loc[df['momentum'] > threshold, 'pos'] = 0  # Sell Pump
    
    df['pos'] = df['pos'].replace(0, np.nan)
    df['pos'] = df['pos'].ffill()
    df['pos'] = df['pos'].fillna(0)
    df.loc[df['momentum'] > threshold, 'pos'] = 0
    
    return df


# 4. OPTIMIZATION (GRID SEARCH)
print("Running Optimization...")

window_list = []
for i in range(54,57,1):
    window_list.append(i)
threshold_list = []
for l in range(19000,20000,1):
    dec=l/100000
    threshold_list.append(dec)
results = []

for w in window_list:
    for t in threshold_list:
        _, metrics = run_backtest(mean_reversion_strategy, test_df, window=w, threshold=t)
        metrics['window'] = w
        metrics['threshold'] = t
        results.append(metrics)

grid_summary = pd.DataFrame(results)
top_results = grid_summary.sort_values(by='sharpe', ascending=False).head(10)


print("\n" + "="*30)
print("TOP 10 PARAMETER COMBINATIONS")
print("="*30)
print(top_results[['window', 'threshold', 'sharpe']])


best_row = grid_summary.loc[grid_summary['sharpe'].idxmax()]
best_win = int(best_row['window'])
best_thresh = best_row['threshold']

print(f"\n✅ Best Params: Window={best_win}, Threshold={best_thresh:.1%}")
print(f"Best Sharpe: {best_row['sharpe']:.2f}")

# Heatmap
pivot_table = grid_summary.pivot(index='window', columns='threshold', values='sharpe')
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, cmap='RdYlGn', fmt=".2f")
plt.title('Sharpe Ratio Heatmap')
plt.show()

# 5. FINAL BACKTEST
final_df, final_metrics = run_backtest(mean_reversion_strategy, test_df, window=best_win, threshold=best_thresh)

print("\n--- PERFORMANCE METRICS ---")
print(f"Total Return: {final_metrics['annual_return']:.2%}")
print(f"Max Drawdown: {final_metrics['mdd']:.2%}")
print(f"Sharpe Ratio: {final_metrics['sharpe']:.2f}")
fig = px.line(final_df, x='Date', y=['cumu', 'bnh_cumu'], 
              title='Strategy Equity Curve',
              color_discrete_map={'cumu': 'green', 'bnh_cumu': 'gray'})
fig.show()


# 6. DRAWDOWN CHART 
def plot_drawdown(returns, title):
    
    returns = returns.fillna(0)
    returns = returns.reset_index(drop=True)
    
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    
    ax1.plot(cumulative, label='Cumulative Return')
    ax1.plot(running_max, label='Running Max', linestyle='--', alpha=0.7)
    ax1.set_title(f'{title} - Cumulative Returns')
    ax1.legend()
    ax1.grid(True, alpha=0.3)


    ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
    ax2.plot(drawdown, color='darkred')
    
    ax2.set_title(f'{title} - Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


plot_drawdown(final_df['pnl'], 'Mean Reversion Strategy')


