import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import yfinance as yf
from portfolio_functions import optimize_custom_portfolio, calculate_portfolio_metrics

# Page configuration
st.set_page_config(
    page_title="Nifty 50 Portfolio Optimizer",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    with open('portfolio_model_data.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

# Function to refresh data
def refresh_data():
    """Fetch latest stock data and re-run predictions"""
    import yfinance as yf
    from datetime import datetime, timedelta
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    import os
    
    # Disable GPU for TensorFlow
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    st.info("🔄 Step 1/6: Fetching latest stock data...")
    progress_bar = st.progress(0)
    
    # Load existing data to get tickers
    with open('portfolio_model_data.pkl', 'rb') as f:
        old_data = pickle.load(f)
    
    nifty50_tickers = old_data['nifty50_tickers']
    
    # Fetch new stock data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    stock_data = {}
    for i, ticker in enumerate(nifty50_tickers):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
            if not data.empty:
                stock_data[ticker] = data
        except:
            pass
        progress_bar.progress((i + 1) / len(nifty50_tickers) * 0.15)
    
    st.info(f"📰 Step 2/6: Fetching news and analyzing sentiment... ({len(stock_data)} stocks)")
    progress_bar.progress(0.15)
    
    # Load FinBERT
    try:
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(repo_id="yiyanghkust/finbert-tone")
        finbert_tokenizer = BertTokenizer.from_pretrained(model_path)
        finbert_model = BertForSequenceClassification.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        finbert_model.to(device)
        finbert_model.eval()
    except:
        st.error("❌ Failed to load FinBERT model")
        return
    
    # Fetch news and calculate sentiment
    def get_sentiment_score(text):
        if not text or len(text.strip()) == 0:
            return 0.0
        inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = finbert_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        neutral = predictions[0][0].item()
        positive = predictions[0][1].item()
        negative = predictions[0][2].item()
        return positive - negative
    
    daily_sentiment = {}
    for ticker in stock_data.keys():
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            if news:
                sentiment_scores = []
                dates = []
                for article in news:
                    title = article.get('title', '')
                    summary = article.get('summary', '')
                    text = f"{title}. {summary}"
                    score = get_sentiment_score(text)
                    sentiment_scores.append(score)
                    pub_date = datetime.fromtimestamp(article.get('providerPublishTime', 0))
                    dates.append(pub_date.date())
                
                if sentiment_scores:
                    sentiment_df = pd.DataFrame({'date': dates, 'sentiment': sentiment_scores})
                    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
                    daily_avg = sentiment_df.groupby('date')['sentiment'].mean()
                    daily_sentiment[ticker] = daily_avg
        except:
            pass
    
    progress_bar.progress(0.30)
    st.info("🔗 Step 3/6: Merging sentiment with stock data...")
    
    # Merge sentiment with stock data
    enhanced_stock_data = {}
    for ticker in stock_data.keys():
        df = stock_data[ticker].copy()
        df.index = pd.to_datetime(df.index)
        
        if ticker in daily_sentiment:
            sentiment_series = daily_sentiment[ticker]
            df['sentiment'] = sentiment_series
            df['sentiment'] = df['sentiment'].fillna(method='ffill').fillna(0)
        else:
            df['sentiment'] = 0.0
        
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df = df.dropna()
        
        enhanced_stock_data[ticker] = df
    
    progress_bar.progress(0.45)
    st.info("🤖 Step 4/6: Training LSTM models... (This will take a while)")
    
    # Prepare and train LSTM
    def prepare_lstm_data(df, lookback=60):
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'sentiment', 
                    'returns', 'volatility', 'MA_20', 'MA_50']
        data = df[features].values
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, 3])
        return np.array(X), np.array(y), scaler
    
    lstm_models = {}
    scalers = {}
    lookback = 60
    
    for i, ticker in enumerate(enhanced_stock_data.keys()):
        try:
            df = enhanced_stock_data[ticker]
            X, y, scaler = prepare_lstm_data(df, lookback=lookback)
            scalers[ticker] = scaler
            
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test), verbose=0)
            
            lstm_models[ticker] = model
            progress_bar.progress(0.45 + (i + 1) / len(enhanced_stock_data) * 0.35)
        except:
            pass
    
    progress_bar.progress(0.80)
    st.info("📊 Step 5/6: Generating predictions...")
    
    # Generate predictions
    future_predictions = {}
    prediction_days = 30
    
    for ticker in lstm_models.keys():
        try:
            model = lstm_models[ticker]
            scaler = scalers[ticker]
            df = enhanced_stock_data[ticker]
            
            recent_data = df[['Open', 'High', 'Low', 'Close', 'Volume', 'sentiment', 
                              'returns', 'volatility', 'MA_20', 'MA_50']].values[-lookback:]
            scaled_recent = scaler.transform(recent_data)
            
            future_preds = []
            current_batch = scaled_recent.reshape(1, lookback, -1)
            
            for i in range(prediction_days):
                pred = model.predict(current_batch, verbose=0)[0, 0]
                future_preds.append(pred)
                new_row = current_batch[0, -1, :].copy()
                new_row[3] = pred
                new_row = new_row.reshape(1, 1, -1)
                current_batch = np.concatenate([current_batch[:, 1:, :], new_row], axis=1)
            
            dummy_array = np.zeros((len(future_preds), 10))
            dummy_array[:, 3] = future_preds
            denormalized = scaler.inverse_transform(dummy_array)
            future_prices = denormalized[:, 3]
            
            future_predictions[ticker] = future_prices
        except:
            pass
    
    progress_bar.progress(0.90)
    st.info("💼 Step 6/6: Calculating portfolio metrics...")
    
    # Calculate portfolio data
    portfolio_data = {}
    for ticker in future_predictions.keys():
        try:
            current_price = float(enhanced_stock_data[ticker]['Close'].iloc[-1])
            predicted_prices = future_predictions[ticker]
            expected_return = float((predicted_prices[-1] - current_price) / current_price)
            historical_returns = enhanced_stock_data[ticker]['returns'].values
            volatility = float(np.std(historical_returns) * np.sqrt(252))
            
            portfolio_data[ticker] = {
                'current_price': current_price,
                'predicted_price': float(predicted_prices[-1]),
                'expected_return': expected_return,
                'volatility': volatility,
                'mean_sentiment': float(enhanced_stock_data[ticker]['sentiment'].mean())
            }
        except:
            pass
    
    # Save updated data
    streamlit_data = {
        'portfolio_data': portfolio_data,
        'enhanced_stock_data': enhanced_stock_data,
        'lstm_models': lstm_models,
        'scalers': scalers,
        'future_predictions': future_predictions,
        'nifty50_tickers': nifty50_tickers,
        'start_date': start_date,
        'end_date': end_date
    }
    
    with open('portfolio_model_data.pkl', 'wb') as f:
        pickle.dump(streamlit_data, f)
    
    progress_bar.progress(1.0)
    st.success("✅ Data refreshed successfully!")
    st.balloons()
    
    # Clear cache and rerun
    st.cache_data.clear()
    st.rerun()

# Main app
def main():
    st.markdown('<h1 class="main-header">📊 Nifty 50 Portfolio Optimizer</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by FinBERT Sentiment Analysis & LSTM Price Prediction")
    
    # Load data
    try:
        data = load_data()
        portfolio_data = data['portfolio_data']
        enhanced_stock_data = data['enhanced_stock_data']
        start_date = data['start_date']
        end_date = data['end_date']
    except FileNotFoundError:
        st.error("❌ Data file not found! Please run the Jupyter notebook first to generate 'portfolio_model_data.pkl'")
        return
    
    # Sidebar
    st.sidebar.title("⚙️ Portfolio Settings")
    st.sidebar.markdown("---")
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Market Data", type="primary"):
        refresh_data()
    
    st.sidebar.markdown(f"**Data Period:** {start_date.date()} to {end_date.date()}")
    st.sidebar.markdown(f"**Available Stocks:** {len(portfolio_data)}")
    st.sidebar.markdown("---")
    
    # Portfolio constraints
    st.sidebar.subheader("📋 Portfolio Constraints")
    
    # Number of stocks slider
    max_stocks = len(portfolio_data)
    num_stocks = st.sidebar.slider(
        "Number of Stocks",
        min_value=3,
        max_value=max_stocks,
        value=min(10, max_stocks),
        help="Select how many stocks to include in your portfolio"
    )
    
    # Target return slider
    use_target_return = st.sidebar.checkbox("Set Target Return", value=False)
    target_return = None
    if use_target_return:
        target_return = st.sidebar.slider(
            "Minimum Expected Return (%)",
            min_value=0.0,
            max_value=50.0,
            value=15.0,
            step=1.0,
            help="Minimum annual return you want to achieve"
        ) / 100
    
    # Sharpe ratio slider
    use_sharpe = st.sidebar.checkbox("Set Minimum Sharpe Ratio", value=False)
    min_sharpe = None
    if use_sharpe:
        min_sharpe = st.sidebar.slider(
            "Minimum Sharpe Ratio",
            min_value=0.0,
            max_value=5.0,
            value=1.5,
            step=0.1,
            help="Higher Sharpe ratio means better risk-adjusted returns"
        )
    
    # Risk tolerance slider
    use_max_risk = st.sidebar.checkbox("Set Maximum Risk", value=False)
    max_risk = None
    if use_max_risk:
        max_risk = st.sidebar.slider(
            "Maximum Risk/Volatility (%)",
            min_value=5.0,
            max_value=50.0,
            value=25.0,
            step=1.0,
            help="Maximum volatility you're willing to accept"
        ) / 100
    
    # Risk-free rate
    st.sidebar.markdown("---")
    risk_free_rate = st.sidebar.number_input(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=6.0,
        step=0.5,
        help="Used for Sharpe ratio calculation (e.g., treasury bill rate)"
    ) / 100
    
    # Optimize button
    st.sidebar.markdown("---")
    optimize_clicked = st.sidebar.button("🎯 Optimize Portfolio", type="primary", use_container_width=True)
    
    # Main content
    if optimize_clicked:
        with st.spinner("🔍 Finding optimal portfolio..."):
            result = optimize_custom_portfolio(
                portfolio_data=portfolio_data,
                enhanced_stock_data=enhanced_stock_data,
                num_stocks=num_stocks,
                target_return=target_return,
                min_sharpe=min_sharpe,
                max_risk=max_risk,
                risk_free_rate=risk_free_rate
            )
        
        if result is None:
            st.error("❌ No portfolios found matching your constraints!")
            st.warning("💡 Try relaxing your requirements (lower target return, higher max risk, etc.)")
            return
        
        # Display results
        st.success(f"✅ Optimal portfolio found! ({result['num_valid_portfolios']:,} valid portfolios out of {result['total_portfolios_tested']:,} tested)")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Expected Return",
                f"{result['expected_return']:.2%}",
                help="Predicted annual return"
            )
        
        with col2:
            st.metric(
                "Risk (Volatility)",
                f"{result['volatility']:.2%}",
                help="Annual volatility"
            )
        
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{result['sharpe_ratio']:.4f}",
                help="Risk-adjusted return measure"
            )
        
        with col4:
            st.metric(
                "Number of Stocks",
                len(result['tickers'])
            )
        
        st.markdown("---")
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio Allocation", "📈 Stock Details", "📉 Risk Analysis", "💰 Investment Calculator"])
        
        with tab1:
            st.subheader("Portfolio Allocation")
            
            # Create allocation dataframe
            allocation_df = pd.DataFrame({
                'Stock': result['tickers'],
                'Weight (%)': [w * 100 for w in result['weights']],
                'Expected Return (%)': [portfolio_data[t]['expected_return'] * 100 for t in result['tickers']],
                'Risk (%)': [portfolio_data[t]['volatility'] * 100 for t in result['tickers']],
                'Current Price': [portfolio_data[t]['current_price'] for t in result['tickers']],
                'Predicted Price': [portfolio_data[t]['predicted_price'] for t in result['tickers']],
                'Sentiment': [portfolio_data[t]['mean_sentiment'] for t in result['tickers']]
            })
            
            # Filter stocks with >1% allocation
            display_df = allocation_df[allocation_df['Weight (%)'] > 1.0].sort_values('Weight (%)', ascending=False)
            
            # Create two columns
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Pie chart
                fig_pie = px.pie(
                    display_df,
                    values='Weight (%)',
                    names='Stock',
                    title='Portfolio Allocation',
                    hole=0.4
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Bar chart
                fig_bar = px.bar(
                    display_df,
                    x='Stock',
                    y='Weight (%)',
                    title='Stock Weights',
                    color='Expected Return (%)',
                    color_continuous_scale='RdYlGn'
                )
                fig_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Display table
            st.dataframe(
                display_df.style.format({
                    'Weight (%)': '{:.2f}%',
                    'Expected Return (%)': '{:.2f}%',
                    'Risk (%)': '{:.2f}%',
                    'Current Price': '₹{:.2f}',
                    'Predicted Price': '₹{:.2f}',
                    'Sentiment': '{:.3f}'
                }).background_gradient(subset=['Expected Return (%)'], cmap='RdYlGn'),
                use_container_width=True
            )
        
        with tab2:
            st.subheader("Individual Stock Details")
            
            # Stock selector
            selected_stock = st.selectbox(
                "Select a stock to view details:",
                result['tickers']
            )
            
            if selected_stock:
                stock_info = portfolio_data[selected_stock]
                stock_weight = result['weights'][result['tickers'].index(selected_stock)] * 100
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Portfolio Weight", f"{stock_weight:.2f}%")
                with col2:
                    st.metric("Expected Return", f"{stock_info['expected_return']:.2%}")
                with col3:
                    st.metric("Current Price", f"₹{stock_info['current_price']:.2f}")
                with col4:
                    st.metric("Predicted Price", f"₹{stock_info['predicted_price']:.2f}")
                
                # Price chart
                stock_data = enhanced_stock_data[selected_stock]
                fig_stock = go.Figure()
                fig_stock.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Close'],
                    mode='lines',
                    name='Historical Price',
                    line=dict(color='blue')
                ))
                fig_stock.update_layout(
                    title=f"{selected_stock} - Historical Price",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    hovermode='x unified'
                )
                st.plotly_chart(fig_stock, use_container_width=True)
        
        with tab3:
            st.subheader("Risk Analysis")
            
            # Risk-Return scatter
            scatter_df = pd.DataFrame({
                'Stock': result['tickers'],
                'Expected Return (%)': [portfolio_data[t]['expected_return'] * 100 for t in result['tickers']],
                'Risk (%)': [portfolio_data[t]['volatility'] * 100 for t in result['tickers']],
                'Weight (%)': [w * 100 for w in result['weights']]
            })
            
            fig_scatter = px.scatter(
                scatter_df,
                x='Risk (%)',
                y='Expected Return (%)',
                size='Weight (%)',
                hover_name='Stock',
                title='Risk vs Return Analysis',
                labels={'Risk (%)': 'Annual Volatility (%)', 'Expected Return (%)': 'Expected Return (%)'}
            )
            fig_scatter.add_hline(
                y=result['expected_return'] * 100,
                line_dash="dash",
                line_color="green",
                annotation_text="Portfolio Return"
            )
            fig_scatter.add_vline(
                x=result['volatility'] * 100,
                line_dash="dash",
                line_color="red",
                annotation_text="Portfolio Risk"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Diversification metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Portfolio Diversification", f"{len(result['tickers'])} stocks")
                st.metric("Largest Position", f"{max(result['weights']) * 100:.2f}%")
            
            with col2:
                st.metric("Smallest Position", f"{min(result['weights']) * 100:.2f}%")
                avg_weight = np.mean(result['weights']) * 100
                st.metric("Average Position", f"{avg_weight:.2f}%")
        
        with tab4:
            st.subheader("Investment Calculator")
            
            investment_amount = st.number_input(
                "Enter your investment amount (₹):",
                min_value=1000.0,
                max_value=10000000.0,
                value=100000.0,
                step=1000.0
            )
            
            if investment_amount:
                st.markdown("### 📋 Investment Breakdown")
                
                # Calculate investment per stock
                investment_df = pd.DataFrame({
                    'Stock': result['tickers'],
                    'Weight (%)': [w * 100 for w in result['weights']],
                    'Investment (₹)': [w * investment_amount for w in result['weights']],
                    'Current Price (₹)': [portfolio_data[t]['current_price'] for t in result['tickers']],
                    'Shares to Buy': [int((w * investment_amount) / portfolio_data[t]['current_price']) 
                                     for w, t in zip(result['weights'], result['tickers'])],
                    'Actual Investment (₹)': [int((w * investment_amount) / portfolio_data[t]['current_price']) * 
                                             portfolio_data[t]['current_price'] 
                                             for w, t in zip(result['weights'], result['tickers'])]
                })
                
                # Filter and sort
                investment_df = investment_df[investment_df['Weight (%)'] > 1.0].sort_values('Investment (₹)', ascending=False)
                
                st.dataframe(
                    investment_df.style.format({
                        'Weight (%)': '{:.2f}%',
                        'Investment (₹)': '₹{:,.2f}',
                        'Current Price (₹)': '₹{:.2f}',
                        'Shares to Buy': '{:.0f}',
                        'Actual Investment (₹)': '₹{:,.2f}'
                    }),
                    use_container_width=True
                )
                
                # Summary
                total_actual = investment_df['Actual Investment (₹)'].sum()
                expected_value = total_actual * (1 + result['expected_return'])
                expected_profit = expected_value - total_actual
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Investment", f"₹{total_actual:,.2f}")
                with col2:
                    st.metric("Expected Value (1 year)", f"₹{expected_value:,.2f}", 
                             delta=f"₹{expected_profit:,.2f}")
                with col3:
                    st.metric("Expected Profit", f"₹{expected_profit:,.2f}", 
                             delta=f"{result['expected_return']:.2%}")
    
    else:
        # Show welcome screen
        st.info("👈 Use the sidebar to set your portfolio preferences and click 'Optimize Portfolio' to get started!")
        
        # Show available stocks
        st.subheader("📋 Available Stocks")
        
        stocks_df = pd.DataFrame({
            'Stock': list(portfolio_data.keys()),
            'Current Price': [portfolio_data[t]['current_price'] for t in portfolio_data.keys()],
            'Expected Return (%)': [portfolio_data[t]['expected_return'] * 100 for t in portfolio_data.keys()],
            'Risk (%)': [portfolio_data[t]['volatility'] * 100 for t in portfolio_data.keys()],
            'Sentiment': [portfolio_data[t]['mean_sentiment'] for t in portfolio_data.keys()]
        }).sort_values('Expected Return (%)', ascending=False)
        
        st.dataframe(
            stocks_df.style.format({
                'Current Price': '₹{:.2f}',
                'Expected Return (%)': '{:.2f}%',
                'Risk (%)': '{:.2f}%',
                'Sentiment': '{:.3f}'
            }).background_gradient(subset=['Expected Return (%)'], cmap='RdYlGn'),
            use_container_width=True,
            height=400
        )

if __name__ == "__main__":
    main()