"""HTML report generation for ml4t.backtest backtests."""

import json
from pathlib import Path
from typing import Any

from ml4t.backtest.portfolio.accounting import PortfolioAccounting
from ml4t.backtest.reporting.base import ReportGenerator


class HTMLReportGenerator(ReportGenerator):
    """
    Generates comprehensive HTML reports for backtest results.

    Creates interactive reports with:
    - Performance summary
    - Equity curve charts
    - Trade analysis
    - Risk metrics
    - Asset class breakdown
    """

    def generate(
        self,
        accounting: PortfolioAccounting,
        strategy_params: dict[str, Any] | None = None,
        backtest_params: dict[str, Any] | None = None,
    ) -> Path:
        """
        Generate HTML report from portfolio accounting data.

        Args:
            accounting: Portfolio accounting with results
            strategy_params: Strategy configuration parameters
            backtest_params: Backtest configuration parameters

        Returns:
            Path to generated HTML report
        """
        # Prepare report data
        report_data = self._prepare_report_data(accounting, strategy_params, backtest_params)

        # Generate HTML content
        html_content = self._generate_html_content(report_data)

        # Save report
        report_path = self.output_dir / f"{self.report_name}.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return report_path

    def _generate_html_content(self, report_data: dict[str, Any]) -> str:
        """Generate the complete HTML report content."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ml4t.backtest Backtest Report - {report_data["metadata"]["report_name"]}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        {self._generate_header(report_data)}
        {self._generate_summary_section(report_data)}
        {self._generate_performance_section(report_data)}
        {self._generate_charts_section(report_data)}
        {self._generate_trades_section(report_data)}
        {self._generate_positions_section(report_data)}
        {self._generate_risk_section(report_data)}
        {self._generate_footer(report_data)}
    </div>

    <script>
        {self._generate_javascript(report_data)}
    </script>
</body>
</html>
"""
        return html

    def _get_css_styles(self) -> str:
        """Get CSS styles for the report."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .header .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .section {
            background: white;
            margin-bottom: 30px;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .section h2 {
            color: #667eea;
            font-size: 1.8em;
            margin-bottom: 20px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }

        .metric-card {
            background: #f8f9ff;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }

        .metric-label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .metric-value {
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }

        .metric-value.positive {
            color: #28a745;
        }

        .metric-value.negative {
            color: #dc3545;
        }

        .chart-container {
            margin: 20px 0;
            min-height: 400px;
        }

        .table-container {
            overflow-x: auto;
            margin-top: 20px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
        }

        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        th {
            background-color: #f8f9ff;
            font-weight: 600;
            color: #667eea;
        }

        tr:hover {
            background-color: #f8f9ff;
        }

        .footer {
            text-align: center;
            color: #666;
            font-size: 0.9em;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }

        .warning {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }

        .info {
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }

        @media (max-width: 768px) {
            .metrics-grid {
                grid-template-columns: 1fr;
            }

            .header h1 {
                font-size: 2em;
            }
        }
        """

    def _generate_header(self, report_data: dict[str, Any]) -> str:
        """Generate the report header."""
        metadata = report_data["metadata"]
        return f"""
        <div class="header">
            <h1>ml4t.backtest Backtest Report</h1>
            <div class="subtitle">
                {metadata["report_name"]}<br>
                Generated: {metadata["generated_at"][:19]}
            </div>
        </div>
        """

    def _generate_summary_section(self, report_data: dict[str, Any]) -> str:
        """Generate the summary metrics section."""
        perf = report_data["performance"]
        portfolio = report_data["portfolio"]
        costs = report_data["costs"]

        return f"""
        <div class="section">
            <h2>Performance Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Total Return</div>
                    <div class="metric-value {"positive" if perf["total_return"] >= 0 else "negative"}">
                        {perf["total_return"]:.2%}
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">Total P&L</div>
                    <div class="metric-value {"positive" if perf["total_pnl"] >= 0 else "negative"}">
                        ${perf["total_pnl"]:,.2f}
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value {"positive" if perf.get("sharpe_ratio", 0) >= 0 else "negative"}">
                        {perf.get("sharpe_ratio", "N/A")}
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value negative">
                        -{perf["max_drawdown"]:.2%}
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value">
                        {perf["win_rate"]:.2%}
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">Total Trades</div>
                    <div class="metric-value">
                        {perf["num_trades"]:,}
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">Final Equity</div>
                    <div class="metric-value">
                        ${portfolio["final_equity"]:,.2f}
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">Total Commission</div>
                    <div class="metric-value negative">
                        ${costs["total_commission"]:,.2f}
                    </div>
                </div>
            </div>
        </div>
        """

    def _generate_performance_section(self, report_data: dict[str, Any]) -> str:
        """Generate detailed performance metrics."""
        perf = report_data["performance"]
        costs = report_data["costs"]

        return f"""
        <div class="section">
            <h2>Detailed Performance</h2>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Realized P&L</div>
                    <div class="metric-value {"positive" if perf["realized_pnl"] >= 0 else "negative"}">
                        ${perf["realized_pnl"]:,.2f}
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">Unrealized P&L</div>
                    <div class="metric-value {"positive" if perf["unrealized_pnl"] >= 0 else "negative"}">
                        ${perf["unrealized_pnl"]:,.2f}
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">Profit Factor</div>
                    <div class="metric-value">
                        {perf["profit_factor"]:.2f}
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">Avg Commission/Trade</div>
                    <div class="metric-value">
                        ${costs["avg_commission_per_trade"]:.2f}
                    </div>
                </div>
            </div>
        </div>
        """

    def _generate_charts_section(self, report_data: dict[str, Any]) -> str:
        """Generate charts section."""
        return """
        <div class="section">
            <h2>Charts</h2>

            <div class="chart-container">
                <div id="equity-curve-chart"></div>
            </div>

            <div class="chart-container">
                <div id="returns-chart"></div>
            </div>
        </div>
        """

    def _generate_trades_section(self, report_data: dict[str, Any]) -> str:
        """Generate trades analysis section."""
        trades_df = report_data.get("trades")

        if trades_df is None or len(trades_df) == 0:
            return """
            <div class="section">
                <h2>Trade Analysis</h2>
                <div class="info">No trades found in this backtest.</div>
            </div>
            """

        # Get first few trades for display
        display_trades = trades_df.head(20).to_dicts()

        trades_html = ""
        for trade in display_trades:
            trades_html += f"""
            <tr>
                <td>{trade["timestamp"]}</td>
                <td>{trade["asset_id"]}</td>
                <td>{trade["side"].upper()}</td>
                <td>{trade["quantity"]:.2f}</td>
                <td>${trade["price"]:.2f}</td>
                <td>${trade["commission"]:.2f}</td>
                <td>${trade["total_cost"]:.2f}</td>
            </tr>
            """

        return f"""
        <div class="section">
            <h2>Trade Analysis</h2>

            <div class="info">
                Showing first 20 trades out of {len(trades_df)} total trades.
            </div>

            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Timestamp</th>
                            <th>Asset</th>
                            <th>Side</th>
                            <th>Quantity</th>
                            <th>Price</th>
                            <th>Commission</th>
                            <th>Total Cost</th>
                        </tr>
                    </thead>
                    <tbody>
                        {trades_html}
                    </tbody>
                </table>
            </div>
        </div>
        """

    def _generate_positions_section(self, report_data: dict[str, Any]) -> str:
        """Generate current positions section."""
        positions_df = report_data.get("positions")

        if positions_df is None or len(positions_df) == 0:
            return """
            <div class="section">
                <h2>Current Positions</h2>
                <div class="info">No open positions at end of backtest.</div>
            </div>
            """

        positions_html = ""
        for pos in positions_df.to_dicts():
            positions_html += f"""
            <tr>
                <td>{pos["asset_id"]}</td>
                <td>{pos["quantity"]:.2f}</td>
                <td>${pos["cost_basis"]:.2f}</td>
                <td>${pos["last_price"]:.2f}</td>
                <td>${pos["market_value"]:.2f}</td>
                <td class="{"positive" if pos["unrealized_pnl"] >= 0 else "negative"}">${pos["unrealized_pnl"]:.2f}</td>
                <td class="{"positive" if pos["total_pnl"] >= 0 else "negative"}">${pos["total_pnl"]:.2f}</td>
            </tr>
            """

        return f"""
        <div class="section">
            <h2>Current Positions</h2>

            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Asset</th>
                            <th>Quantity</th>
                            <th>Cost Basis</th>
                            <th>Last Price</th>
                            <th>Market Value</th>
                            <th>Unrealized P&L</th>
                            <th>Total P&L</th>
                        </tr>
                    </thead>
                    <tbody>
                        {positions_html}
                    </tbody>
                </table>
            </div>
        </div>
        """

    def _generate_risk_section(self, report_data: dict[str, Any]) -> str:
        """Generate risk metrics section."""
        risk = report_data.get("risk", {})

        return f"""
        <div class="section">
            <h2>Risk Metrics</h2>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Max Leverage</div>
                    <div class="metric-value">
                        {risk.get("max_leverage", 1.0):.2f}x
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">Max Concentration</div>
                    <div class="metric-value">
                        {risk.get("max_concentration", 0.0):.2%}
                    </div>
                </div>
            </div>

            <div class="warning">
                <strong>Risk Disclaimer:</strong> Past performance does not guarantee future results.
                All trading involves risk of loss.
            </div>
        </div>
        """

    def _generate_footer(self, report_data: dict[str, Any]) -> str:
        """Generate report footer."""
        return f"""
        <div class="footer">
            Report generated by ml4t.backtest Backtesting Framework<br>
            Generated at: {report_data["metadata"]["generated_at"]}
        </div>
        """

    def _generate_javascript(self, report_data: dict[str, Any]) -> str:
        """Generate JavaScript for interactive charts."""
        equity_df = report_data.get("equity_curve")

        if equity_df is None or len(equity_df) == 0:
            return "// No data available for charts"

        # Convert DataFrame to JSON for JavaScript
        equity_data = equity_df.to_dicts()

        # Convert datetime objects to strings for JSON serialization
        for item in equity_data:
            if "timestamp" in item:
                item["timestamp"] = item["timestamp"].isoformat()

        return f"""
        // Equity curve chart
        const equityData = {json.dumps(equity_data)};

        const equityTrace = {{
            x: equityData.map(d => d.timestamp),
            y: equityData.map(d => d.equity),
            type: 'scatter',
            mode: 'lines',
            name: 'Equity',
            line: {{
                color: '#667eea',
                width: 2
            }}
        }};

        const equityLayout = {{
            title: 'Equity Curve',
            xaxis: {{ title: 'Date' }},
            yaxis: {{ title: 'Portfolio Value ($)' }},
            margin: {{ t: 50 }}
        }};

        Plotly.newPlot('equity-curve-chart', [equityTrace], equityLayout);

        // Returns chart
        const returnsTrace = {{
            x: equityData.map(d => d.timestamp).slice(1),
            y: equityData.map(d => d.returns).slice(1),
            type: 'scatter',
            mode: 'markers',
            name: 'Daily Returns',
            marker: {{
                color: equityData.map(d => d.returns > 0 ? '#28a745' : '#dc3545').slice(1),
                size: 4
            }}
        }};

        const returnsLayout = {{
            title: 'Daily Returns Distribution',
            xaxis: {{ title: 'Date' }},
            yaxis: {{ title: 'Daily Return' }},
            margin: {{ t: 50 }}
        }};

        Plotly.newPlot('returns-chart', [returnsTrace], returnsLayout);
        """
