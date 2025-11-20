# Bitcoin Mean Reversion Strategy 📈

A quantitative trading strategy backtested on Bitcoin data (2023-2025). This project implements a momentum-based Mean Reversion algorithm optimized using Grid Search to identify profitable entry/exit thresholds.

### 🚀 Project Overview
*   **Objective:** Outperform the Buy & Hold benchmark in a bearish/choppy market regime (2025).
*   **Methodology:** Developed a custom vectorised backtesting engine in Python to simulate trading logic.
*   **Optimization:** Performed a parameter sweep (Window Size vs. Threshold %) to maximize Sharpe Ratio.

### 📊 Key Results (2025 Test Set)
| Metric | Benchmark (Buy & Hold) | Mean Reversion Strategy |
| :--- | :--- | :--- |
| **Total Return** | -3.21% | **+34.88%** |
| **Sharpe Ratio** | Negative | **1.07** |
| **Max Drawdown** | 16.85% | 46.94% |

### 🛠️ Technical Implementation
*   **Data Processing:** `Pandas` for time-series manipulation and cleaning.
*   **Strategy Logic:** 
    *   Calculated rolling percentage changes to detect oversold (Crash) and overbought (Pump) signals.
    *   Implemented `Window=54 Days` and `Threshold=19%` based on heat-map optimization.
*   **Visualization:** `Seaborn` for Heatmaps and `Plotly` for interactive equity curves.
