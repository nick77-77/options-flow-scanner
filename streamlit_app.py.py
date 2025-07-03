<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Options Flow Tracker</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .header h1 {
            font-size: 2.5em;
            background: linear-gradient(45deg, #667eea, #764ba2);
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }

        .header p {
            font-size: 1.2em;
            color: #666;
            margin-bottom: 20px;
        }

        .control-panel {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .control-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }

        .control-group {
            display: flex;
            flex-direction: column;
        }

        .control-group label {
            font-weight: 600;
            margin-bottom: 8px;
            color: #555;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .control-group select,
        .control-group input {
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 14px;
            transition: all 0.3s ease;
            background: white;
        }

        .control-group select:focus,
        .control-group input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        .button-group {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }

        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 10px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
            text-decoration: none;
            color: white;
        }

        .btn-primary {
            background: linear-gradient(45deg, #667eea, #764ba2);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }

        .btn-secondary {
            background: linear-gradient(45deg, #fd746c, #ff9068);
            box-shadow: 0 4px 15px rgba(253, 116, 108, 0.3);
        }

        .btn-secondary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(253, 116, 108, 0.4);
        }

        .btn-success {
            background: linear-gradient(45deg, #4ecdc4, #44a08d);
            box-shadow: 0 4px 15px rgba(78, 205, 196, 0.3);
        }

        .btn-success:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(78, 205, 196, 0.4);
        }

        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease;
        }

        .card:hover {
            transform: translateY(-5px);
        }

        .card h3 {
            margin-bottom: 15px;
            color: #333;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .metric-card {
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            border-radius: 20px 20px 0 0;
        }

        .metric-value {
            font-size: 2.5em;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 10px;
        }

        .metric-label {
            font-size: 1.1em;
            color: #666;
            margin-bottom: 5px;
        }

        .metric-change {
            font-size: 0.9em;
            padding: 4px 8px;
            border-radius: 6px;
            font-weight: 600;
        }

        .positive {
            background: rgba(78, 205, 196, 0.1);
            color: #4ecdc4;
        }

        .negative {
            background: rgba(253, 116, 108, 0.1);
            color: #fd746c;
        }

        .chart-container {
            position: relative;
            height: 300px;
            margin-top: 20px;
        }

        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }

        .data-table th,
        .data-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }

        .data-table th {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            font-weight: 600;
        }

        .data-table tr:hover {
            background: rgba(102, 126, 234, 0.05);
        }

        .alert {
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 4px solid;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .alert-critical {
            background: rgba(253, 116, 108, 0.1);
            border-color: #fd746c;
            color: #d63031;
        }

        .alert-high {
            background: rgba(255, 144, 104, 0.1);
            border-color: #ff9068;
            color: #e17055;
        }

        .alert-info {
            background: rgba(102, 126, 234, 0.1);
            border-color: #667eea;
            color: #6c5ce7;
        }

        .loading {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 50px;
            font-size: 1.2em;
            color: #666;
        }

        .loading::before {
            content: '';
            width: 30px;
            height: 30px;
            border: 3px solid #e0e0e0;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .tabs {
            display: flex;
            margin-bottom: 20px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 10px;
            padding: 5px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }

        .tab {
            flex: 1;
            padding: 12px;
            border: none;
            background: transparent;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.3s ease;
            font-weight: 600;
        }

        .tab.active {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }

        .status-bullish {
            background: #4ecdc4;
            box-shadow: 0 0 10px rgba(78, 205, 196, 0.5);
        }

        .status-bearish {
            background: #fd746c;
            box-shadow: 0 0 10px rgba(253, 116, 108, 0.5);
        }

        .status-neutral {
            background: #ddd;
        }

        @media (max-width: 768px) {
            .control-grid {
                grid-template-columns: 1fr;
            }
            
            .dashboard {
                grid-template-columns: 1fr;
            }
            
            .button-group {
                flex-direction: column;
            }
            
            .tabs {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-rocket"></i> Enhanced Options Flow Tracker</h1>
            <p>Advanced real-time options activity with buy/sell detection, IV analysis, and institutional pattern recognition</p>
        </div>

        <div class="control-panel">
            <div class="control-grid">
                <div class="control-group">
                    <label><i class="fas fa-chart-line"></i> Analysis Type</label>
                    <select id="scanType">
                        <option value="comprehensive">📊 Comprehensive Dashboard</option>
                        <option value="selling">💰 Selling Activity Analysis</option>
                        <option value="iv">📈 IV Analysis</option>
                        <option value="alerts">⚡ Real-time Alerts</option>
                        <option value="dte">🎯 DTE Strategy Analysis</option>
                        <option value="institutional">🏛️ Institutional Flow</option>
                        <option value="gamma">📈 Gamma Squeeze Scanner</option>
                    </select>
                </div>

                <div class="control-group">
                    <label><i class="fas fa-dollar-sign"></i> Premium Range</label>
                    <select id="premiumRange">
                        <option value="all">All Premiums (No Filter)</option>
                        <option value="under100k">Under $100K</option>
                        <option value="under250k">Under $250K</option>
                        <option value="100k-250k">$100K - $250K</option>
                        <option value="250k-500k">$250K - $500K</option>
                        <option value="above250k">Above $250K</option>
                        <option value="above500k">Above $500K</option>
                        <option value="above1m">Above $1M</option>
                    </select>
                </div>

                <div class="control-group">
                    <label><i class="fas fa-calendar-alt"></i> Time to Expiry</label>
                    <select id="dteFilter">
                        <option value="all">All DTE</option>
                        <option value="0dte">0DTE Only</option>
                        <option value="weekly">Weekly (≤7d)</option>
                        <option value="monthly">Monthly (≤30d)</option>
                        <option value="quarterly">Quarterly (≤90d)</option>
                    </select>
                </div>

                <div class="control-group">
                    <label><i class="fas fa-wave-square"></i> Implied Volatility</label>
                    <select id="ivFilter">
                        <option value="all">All IV</option>
                        <option value="low">Low IV (≤20%)</option>
                        <option value="medium">Medium IV (20-40%)</option>
                        <option value="high">High IV (>40%)</option>
                        <option value="extreme">Extreme IV (>50%)</option>
                    </select>
                </div>
            </div>

            <div class="button-group">
                <button class="btn btn-primary" onclick="runScan()">
                    <i class="fas fa-search"></i> Run Enhanced Scan
                </button>
                <button class="btn btn-secondary" onclick="quickScan('0dte')">
                    <i class="fas fa-bolt"></i> Quick 0DTE Scan
                </button>
                <button class="btn btn-success" onclick="quickScan('highiv')">
                    <i class="fas fa-fire"></i> High IV Scan
                </button>
            </div>
        </div>

        <div id="dashboard" class="dashboard" style="display: none;">
            <div class="card metric-card">
                <h3><i class="fas fa-chart-line"></i> Net Flow</h3>
                <div class="metric-value" id="netFlow">$0</div>
                <div class="metric-label">Direction</div>
                <div class="metric-change" id="flowDirection">
                    <span class="status-indicator status-neutral"></span>
                    Neutral
                </div>
            </div>

            <div class="card metric-card">
                <h3><i class="fas fa-balance-scale"></i> Put/Call Ratio</h3>
                <div class="metric-value" id="putCallRatio">0.00</div>
                <div class="metric-label">Premium Based</div>
                <div class="metric-change" id="pcrChange">--</div>
            </div>

            <div class="card metric-card">
                <h3><i class="fas fa-wave-square"></i> Average IV</h3>
                <div class="metric-value" id="avgIV">0.0%</div>
                <div class="metric-label">Market Volatility</div>
                <div class="metric-change" id="ivChange">--</div>
            </div>

            <div class="card metric-card">
                <h3><i class="fas fa-fire"></i> High IV Premium</h3>
                <div class="metric-value" id="highIVPremium">$0</div>
                <div class="metric-label">Premium in >40% IV</div>
                <div class="metric-change" id="highIVChange">--</div>
            </div>

            <div class="card">
                <h3><i class="fas fa-chart-bar"></i> Flow Breakdown</h3>
                <div class="chart-container">
                    <canvas id="flowChart"></canvas>
                </div>
            </div>

            <div class="card">
                <h3><i class="fas fa-chart-pie"></i> IV Distribution</h3>
                <div class="chart-container">
                    <canvas id="ivChart"></canvas>
                </div>
            </div>
        </div>

        <div id="alertsContainer" style="display: none;">
            <div class="card">
                <h3><i class="fas fa-bell"></i> Real-time Alerts</h3>
                <div id="alertsList"></div>
            </div>
        </div>

        <div id="dataContainer" style="display: none;">
            <div class="card">
                <div class="tabs">
                    <button class="tab active" onclick="showTab('calls')">
                        <i class="fas fa-arrow-up"></i> Call Activity
                    </button>
                    <button class="tab" onclick="showTab('puts')">
                        <i class="fas fa-arrow-down"></i> Put Activity
                    </button>
                    <button class="tab" onclick="showTab('premium')">
                        <i class="fas fa-gem"></i> High Premium
                    </button>
                    <button class="tab" onclick="showTab('volume')">
                        <i class="fas fa-volume-up"></i> High Volume
                    </button>
                </div>

                <div id="calls" class="tab-content active">
                    <table class="data-table" id="callsTable">
                        <thead>
                            <tr>
                                <th>Ticker</th>
                                <th>Strike</th>
                                <th>Price</th>
                                <th>IV</th>
                                <th>DTE</th>
                                <th>Premium</th>
                                <th>Side</th>
                                <th>Volume</th>
                                <th>Strategy</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>

                <div id="puts" class="tab-content">
                    <table class="data-table" id="putsTable">
                        <thead>
                            <tr>
                                <th>Ticker</th>
                                <th>Strike</th>
                                <th>Price</th>
                                <th>IV</th>
                                <th>DTE</th>
                                <th>Premium</th>
                                <th>Side</th>
                                <th>Volume</th>
                                <th>Strategy</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>

                <div id="premium" class="tab-content">
                    <table class="data-table" id="premiumTable">
                        <thead>
                            <tr>
                                <th>Ticker</th>
                                <th>Type</th>
                                <th>Strike</th>
                                <th>Price</th>
                                <th>IV</th>
                                <th>Premium</th>
                                <th>Side</th>
                                <th>Volume</th>
                                <th>Strategy</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>

                <div id="volume" class="tab-content">
                    <table class="data-table" id="volumeTable">
                        <thead>
                            <tr>
                                <th>Ticker</th>
                                <th>Type</th>
                                <th>Price</th>
                                <th>IV</th>
                                <th>Volume</th>
                                <th>Premium</th>
                                <th>Vol/OI</th>
                                <th>Strategy</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </div>

        <div id="loadingContainer" class="loading" style="display: none;">
            Loading enhanced options data...
        </div>
    </div>

    <script>
        // Global variables
        let currentData = [];
        let flowChart = null;
        let ivChart = null;

        // Configuration
        const config = {
            UW_TOKEN: 'e6e8601a-0746-4cec-a07d-c3eabfc13926',
            EXCLUDE_TICKERS: ['TSLA', 'MSTR', 'CRCL'],
            ALLOWED_TICKERS: ['QQQ', 'SPY', 'IWM'],
            MIN_PREMIUM: 50000,
            LIMIT: 1000,
            HIGH_IV_THRESHOLD: 0.30,
            IV_CRUSH_THRESHOLD: 0.15
        };

        // Mock data generator for demonstration
        function generateMockData() {
            const tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'];
            const types = ['C', 'P'];
            const sides = ['BUY_TO_OPEN', 'SELL_TO_OPEN', 'BUY_TO_CLOSE', 'SELL_TO_CLOSE'];
            
            const data = [];
            for (let i = 0; i < 50; i++) {
                const ticker = tickers[Math.floor(Math.random() * tickers.length)];
                const type = types[Math.floor(Math.random() * types.length)];
                const strike = Math.round(Math.random() * 200 + 100);
                const premium = Math.round(Math.random() * 500000 + 10000);
                const iv = Math.random() * 0.6 + 0.1;
                const volume = Math.round(Math.random() * 1000 + 10);
                const dte = Math.round(Math.random() * 90);
                
                data.push({
                    ticker,
                    type,
                    strike,
                    premium,
                    iv,
                    volume,
                    dte,
                    order_side: sides[Math.floor(Math.random() * sides.length)],
                    price: Math.random() * 10 + 0.5,
                    scenarios: ['Normal Flow', 'High Volume', 'Institutional Flow'][Math.floor(Math.random() * 3)],
                    time_ny: new Date().toLocaleTimeString()
                });
            }
            return data;
        }

        // Filter functions
        function applyFilters(data) {
            const premiumRange = document.getElementById('premiumRange').value;
            const dteFilter = document.getElementById('dteFilter').value;
            const ivFilter = document.getElementById('ivFilter').value;

            return data.filter(trade => {
                // Premium filter
                if (!applyPremiumFilter(trade.premium, premiumRange)) return false;
                
                // DTE filter
                if (!applyDteFilter(trade.dte, dteFilter)) return false;
                
                // IV filter
                if (!applyIvFilter(trade.iv, ivFilter)) return false;
                
                return true;
            });
        }

        function applyPremiumFilter(premium, range) {
            switch(range) {
                case 'all': return true;
                case 'under100k': return premium < 100000;
                case 'under250k': return premium < 250000;
                case '100k-250k': return premium >= 100000 && premium < 250000;
                case '250k-500k': return premium >= 250000 && premium < 500000;
                case 'above250k': return premium >= 250000;
                case 'above500k': return premium >= 500000;
                case 'above1m': return premium >= 1000000;
                default: return true;
            }
        }

        function applyDteFilter(dte, filter) {
            switch(filter) {
                case 'all': return true;
                case '0dte': return dte === 0;
                case 'weekly': return dte <= 7;
                case 'monthly': return dte <= 30;
                case 'quarterly': return dte <= 90;
                default: return true;
            }
        }

        function applyIvFilter(iv, filter) {
            switch(filter) {
                case 'all': return true;
                case 'low': return iv <= 0.20;
                case 'medium': return iv > 0.20 && iv <= 0.40;
                case 'high': return iv > 0.40;
                case 'extreme': return iv > 0.50;
                default: return true;
            }
        }

        // Calculate metrics
        function calculateMetrics(trades) {
            const callTrades = trades.filter(t => t.type === 'C');
            const putTrades = trades.filter(t => t.type === 'P');
            
            const callPremiumBought = callTrades.filter(t => t.order_side.includes('BUY')).reduce((sum, t) => sum + t.premium, 0);
            const callPremiumSold = callTrades.filter(t => t.order_side.includes('SELL')).reduce((sum, t) => sum + t.premium, 0);
            const putPremiumBought = putTrades.filter(t => t.order_side.includes('BUY')).reduce((sum, t) => sum + t.premium, 0);
            const putPremiumSold = putTrades.filter(t => t.order_side.includes('SELL')).reduce((sum, t) => sum + t.premium, 0);
            
            const netCallFlow = callPremiumBought - callPremiumSold;
            const netPutFlow = putPremiumBought - putPremiumSold;
            const netTotalFlow = netCallFlow + netPutFlow;
            
            const putCallRatio = (putPremiumBought + putPremiumSold) / Math.max(callPremiumBought + callPremiumSold, 1);
            const avgIV = trades.reduce((sum, t) => sum + t.iv, 0) / trades.length;
            const highIVPremium = trades.filter(t => t.iv > config.HIGH_IV_THRESHOLD).reduce((sum, t) => sum + t.premium, 0);
            
            return {
                netTotalFlow,
                putCallRatio,
                avgIV,
                highIVPremium,
                callPremiumBought,
                callPremiumSold,
                putPremiumBought,
                putPremiumSold
            };
        }

        // Update dashboard
        function updateDashboard(trades) {
            const metrics = calculateMetrics(trades);
            
            // Update metric cards
            document.getElementById('netFlow').textContent = formatCurrency(metrics.netTotalFlow);
            document.getElementById('putCallRatio').textContent = metrics.putCallRatio.toFixed(2);
            document.getElementById('avgIV').textContent = (metrics.avgIV * 100).toFixed(1) + '%';
            document.getElementById('highIVPremium').textContent = formatCurrency(metrics.highIVPremium);
            
            // Update flow direction
            const flowDirection = document.getElementById('flowDirection');
            const indicator = flowDirection.querySelector('.status-indicator');
            
            if (metrics.netTotalFlow > 0) {
                flowDirection.innerHTML = '<span class="status-indicator status-bullish"></span>Bullish';
                flowDirection.className = 'metric-change positive';
            } else if (metrics.netTotalFlow < 0) {
                flowDirection.innerHTML = '<span class="status-indicator status-bearish"></span>Bearish';
                flowDirection.className = 'metric-change negative';
            } else {
                flowDirection.innerHTML = '<span class="status-indicator status-neutral"></span>Neutral';
                flowDirection.className = 'metric-change';
            }
            
            // Update charts
            updateFlowChart(metrics);
            updateIVChart(trades);
        }

        // Update flow chart
        function updateFlowChart(metrics) {
            const ctx = document.getElementById('flowChart').getContext('2d');
            
            if (flowChart) {
                flowChart.destroy();
            }
            
            flowChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Calls Bought', 'Calls Sold', 'Puts Bought', 'Puts Sold'],
                    datasets: [{
                        data: [
                            metrics.callPremiumBought,
                            metrics.callPremiumSold,
                            metrics.putPremiumBought,
                            metrics.putPremiumSold
                        ],
                        backgroundColor: [
                            'rgba(78, 205, 196, 0.8)',
                            'rgba(253, 116, 108, 0.8)',
                            'rgba(255, 107, 107, 0.8)',
                            'rgba(107, 107, 255, 0.8)'
                        ],
                        borderColor: [
                            'rgba(78, 205, 196, 1)',
                            'rgba(253, 116, 108, 1)',
                            'rgba(255, 107, 107, 1)',
                            'rgba(107, 107, 255, 1)'
                        ],
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                callback: function(value) {
                                    return formatCurrency(value);
                                }
                            }
                        }
                    }
                }
            });
        }

        // Update IV chart
        function updateIVChart(trades) {
            const ctx = document.getElementById('ivChart').getContext('2d');
            
            if (ivChart) {
                ivChart.destroy();
            }
            
            const lowIV = trades.filter(t => t.iv <= 0.20).length;
            const mediumIV = trades.filter(t => t.iv > 0.20 && t.iv <= 0.40).length;
            const highIV = trades.filter(t => t.iv > 0.40).length;
            
            ivChart = new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: ['Low IV (≤20%)', 'Medium IV (20-40%)', 'High IV (>40%)'],
                    datasets: [{
                        data: [lowIV, mediumIV, highIV],
                        backgroundColor: [
                            'rgba(144, 238, 144, 0.8)',
                            'rgba(255, 215, 0, 0.8)',
                            'rgba(255, 107, 107, 0.8)'
                        ],
                        borderColor: [
                            'rgba(144, 238, 144, 1)',
                            'rgba(255, 215, 0, 1)',
                            'rgba(255, 107, 107, 1)'
                        ],
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });
        }

        // Update data tables
        function updateDataTables(trades) {
            const callTrades = trades.filter(t => t.type === 'C').sort((a, b) => b.premium - a.premium);
            const putTrades = trades.filter(t => t.type === 'P').sort((a, b) => b.premium - a.premium);
            const premiumTrades = trades.sort((a, b) => b.premium - a.premium);
            const volumeTrades = trades.sort((a, b) => b.volume - a.volume);
            
            populateTable('callsTable', callTrades.slice(0, 20));
            populateTable('putsTable', putTrades.slice(0, 20));
            populateTable('premiumTable', premiumTrades.slice(0, 20));
            populateTable('volumeTable', volumeTrades.slice(0, 20));
        }

        function populateTable(tableId, trades) {
            const tbody = document.querySelector(`#${tableId} tbody`);
            tbody.innerHTML = '';
            
            trades.forEach(trade => {
                const row = tbody.insertRow();
                const cells = [
                    trade.ticker,
                    `$${trade.strike}`,
                    `$${trade.price.toFixed(2)}`,
                    `${(trade.iv * 100).toFixed(1)}%`,
                    tableId === 'volumeTable' ? trade.volume : trade.dte,
                    formatCurrency(trade.premium),
                    tableId === 'volumeTable' ? `${(trade.volume / Math.max(trade.oi || 1, 1)).toFixed(1)}` : trade.order_side,
                    tableId === 'volumeTable' ? trade.scenarios : trade.volume,
                    tableId === 'volumeTable' ? '' : trade.scenarios
                ];
                
                cells.forEach((cellData, index) => {
                    if (cellData !== '') {
                        const cell = row.insertCell(index);
                        cell.textContent = cellData;
                    }
                });
            });
        }

        // Generate alerts
        function generateAlerts(trades) {
            const alerts = [];
            
            trades.forEach(trade => {
                let score = 0;
                let reasons = [];
                let alertType = 'info';
                
                // Premium scoring
                if (trade.premium > 1000000) {
                    score += 5;
                    reasons.push('Massive Premium (>$1M)');
                    alertType = 'critical';
                } else if (trade.premium > 500000) {
                    score += 3;
                    reasons.push('Large Premium (>$500K)');
                    alertType = 'high';
                }
                
                // IV scoring
                if (trade.iv > 0.50) {
                    score += 3;
                    reasons.push('Extreme IV (>50%)');
                    alertType = 'critical';
                } else if (trade.iv > config.HIGH_IV_THRESHOLD) {
                    score += 2;
                    reasons.push('High IV (>30%)');
                }
                
                // Volume scoring
                const volOIRatio = trade.volume / Math.max(trade.oi || 1, 1);
                if (volOIRatio > 5) {
                    score += 3;
                    reasons.push('Extremely High Vol/OI');
                }
                
                if (score >= 4) {
                    alerts.push({
                        ...trade,
                        score,
                        reasons,
                        alertType
                    });
                }
            });
            
            return alerts.sort((a, b) => b.score - a.score);
        }

        function displayAlerts(alerts) {
            const alertsList = document.getElementById('alertsList');
            alertsList.innerHTML = '';
            
            if (alerts.length === 0) {
                alertsList.innerHTML = '<p>No alerts triggered with current criteria.</p>';
                return;
            }
            
            alerts.slice(0, 10).forEach((alert, index) => {
                const alertDiv = document.createElement('div');
                alertDiv.className = `alert alert-${alert.alertType}`;
                
                const icon = alert.alertType === 'critical' ? '🔥' : 
                           alert.alertType === 'high' ? '⚠️' : 'ℹ️';
                
                alertDiv.innerHTML = `
                    <div>
                        <strong>${icon} ${index + 1}. ${alert.ticker} $${alert.strike}${alert.type} (${alert.dte}d)</strong>
                        <br>
                        💰 Premium: ${formatCurrency(alert.premium)} | Price: $${alert.price.toFixed(2)} | IV: ${(alert.iv * 100).toFixed(1)}% | Side: ${alert.order_side}
                        <br>
                        📍 Reasons: ${alert.reasons.join(', ')}
                        <br>
                        <small>Score: ${alert.score} | Strategy: ${alert.scenarios}</small>
                    </div>
                `;
                
                alertsList.appendChild(alertDiv);
            });
        }

        // Utility functions
        function formatCurrency(amount) {
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD',
                minimumFractionDigits: 0,
                maximumFractionDigits: 0
            }).format(amount);
        }

        function showLoading() {
            document.getElementById('loadingContainer').style.display = 'block';
            document.getElementById('dashboard').style.display = 'none';
            document.getElementById('alertsContainer').style.display = 'none';
            document.getElementById('dataContainer').style.display = 'none';
        }

        function hideLoading() {
            document.getElementById('loadingContainer').style.display = 'none';
        }

        function showTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab content
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }

        // Main scan function
        function runScan() {
            showLoading();
            
            // Simulate API call with mock data
            setTimeout(() => {
                const mockData = generateMockData();
                const filteredData = applyFilters(mockData);
                currentData = filteredData;
                
                const scanType = document.getElementById('scanType').value;
                
                hideLoading();
                
                if (scanType === 'comprehensive') {
                    document.getElementById('dashboard').style.display = 'grid';
                    document.getElementById('dataContainer').style.display = 'block';
                    updateDashboard(filteredData);
                    updateDataTables(filteredData);
                } else if (scanType === 'alerts') {
                    document.getElementById('alertsContainer').style.display = 'block';
                    const alerts = generateAlerts(filteredData);
                    displayAlerts(alerts);
                } else {
                    document.getElementById('dashboard').style.display = 'grid';
                    updateDashboard(filteredData);
                }
            }, 2000);
        }

        function quickScan(type) {
            if (type === '0dte') {
                document.getElementById('dteFilter').value = '0dte';
                document.getElementById('premiumRange').value = 'all';
            } else if (type === 'highiv') {
                document.getElementById('ivFilter').value = 'high';
            }
            runScan();
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            // Set default values
            document.getElementById('premiumRange').value = 'under100k';
            
            // Add event listeners for real-time filtering
            document.getElementById('scanType').addEventListener('change', function() {
                if (currentData.length > 0) {
                    runScan();
                }
            });
        });
    </script>
</body>
</html>
