from flask import Flask, render_template, request, jsonify, url_for, redirect
import requests
from binance.client import Client as BinanceClient
from kucoin.client import Market as KucoinMarket
import yfinance as yf
import uuid
import redis
import json
from celery import Celery
import os
import logging
import pandas as pd
import numpy as np
from scipy import stats
import traceback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib as plt
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from keras import Model, Input
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Bidirectional, Dropout, Dense, Conv1D, MaxPooling1D, LayerNormalization, \
    MultiHeadAttention, Add, GlobalAveragePooling1D, TimeDistributed
from keras.src.optimizers import Adam
from statsmodels.tsa.stattools import coint
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Disable GPU for TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def datetimeformat(value, format='%Y'):
    if value == 'now':
        return datetime.now().strftime(format)
    elif isinstance(value, datetime):
        return value.strftime(format)
    return value  # Fallback: return as-is if not a date

app.jinja_env.filters['zip'] = zip
app.jinja_env.filters['datetimeformat'] = datetimeformat

# Use Render's PORT environment variable
port = int(os.environ.get('PORT', 5000))

# Get Redis URL from environment variable or use default
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Set up Redis connection without SSL
try:
    r = redis.Redis.from_url(redis_url)
    r.ping()  # Test connection
    logger = logging.getLogger(__name__)
    logger.info(f"Successfully connected to Redis at {redis_url}")
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to connect to Redis: {str(e)}")

# Celery configuration without SSL
app.config.update(
    CELERY_BROKER_URL=redis_url,
    CELERY_RESULT_BACKEND=redis_url,
    BROKER_CONNECTION_RETRY=True,
    BROKER_CONNECTION_RETRY_ON_STARTUP=True
)

# Initialize Celery
celery = Celery(app.name, broker=redis_url, backend=redis_url)
celery.conf.update(app.config)

# Binance API client - Use environment variables for credentials
binance_api_key = os.getenv('BINANCE_API_KEY', 'enter_key')
binance_api_secret = os.getenv('BINANCE_API_SECRET', 'enter_pass')
binance_client = BinanceClient(binance_api_key, binance_api_secret)

# KuCoin API client
kucoin_api_key = os.getenv('KUCOIN_API_KEY', 'enter_key')
kucoin_api_secret = os.getenv('KUCOIN_API_SECRET', 'enter_pass')
kucoin_api_passphrase = os.getenv('KUCOIN_API_PASSPHRASE', 'enter_passphrase')
kucoin_client = KucoinMarket(kucoin_api_key, kucoin_api_secret, kucoin_api_passphrase)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Directories
DATA_DIR = os.path.join(os.path.dirname(__file__), 'simulations')
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
TASK_DIR = os.path.join(os.path.dirname(__file__), 'tasks')
for directory in [DATA_DIR, STATIC_DIR, TASK_DIR]:
    os.makedirs(directory, exist_ok=True)

# Set temporary directory for appdirs
import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"

@celery.task(bind=True)
def process_portfolio_task(self, cryptos, allocations, start_date, end_date, initial_capital, portfolio_index, task_id,
                           ml_model='LSTM'):
    import tensorflow as tf
    print(f"Processing task for portfolio {portfolio_index} with cryptos {cryptos}")
    logger.info(f"Starting background processing for portfolio {portfolio_index} with model {ml_model}")
    task_status_path = os.path.join(TASK_DIR, f"{task_id}.json")
    filename = f"{ml_model.lower()}_prediction_{portfolio_index}.html"

    update_task_status(task_id, {
        'status': 'PROCESSING',
        'progress': 0,
        'message': 'Starting portfolio simulation...',
        'portfolio_index': portfolio_index,
        'cryptos': cryptos,
        'allocations': dict(zip(cryptos, allocations)),  # Convert list to dict
        'start_date': start_date,
        'capital': str(initial_capital),
        'frequency': 'N/A',
        'end_date': end_date,
        'initial_capital': float(initial_capital),
        'result': None,
        'error': None
    })

    try:
        update_task_status(task_id, {
            'progress': 10,
            'message': 'Fetching historical data...'
        })

        result = get_historical_data(cryptos, allocations, start_date, end_date, initial_capital)
        if result[0] is None:
            raise ValueError(f"No valid data retrieved for portfolio {portfolio_index + 1}")

        total_holdings, detailed_portfolio, correlation_matrix, metrics, crypto_prices, _ = result
        final_value = total_holdings.iloc[-1]
        performance = ((final_value - initial_capital) / initial_capital) * 100

        update_task_status(task_id, {
            'progress': 30,
            'message': f'Generating {ml_model} predictions...',
            'initial_capital': initial_capital,
            'cryptos': cryptos
        })

        # Clear TensorFlow session to release memory
        tf.keras.backend.clear_session()

        # Prediction logic with forecast_days=252
        predictions_dict = {}
        for i, crypto in enumerate(cryptos):
            update_task_status(task_id, {
                'message': f'Training {ml_model} model for {crypto} ({i + 1}/{len(cryptos)})...'
            })
            preds = select_model_for_prediction(crypto_prices[crypto], ml_model, lookback=60, forecast_days=252)
            logger.debug(f"Predictions for {crypto}: Expected 252, got {len(preds)}")
            predictions_dict[crypto] = preds if preds is not None else np.zeros(252)
            update_task_status(task_id, {
                'progress': 30 + int(40 * (i + 1) / len(cryptos)),
            })

        predicted_holdings = pd.DataFrame(index=range(252), columns=cryptos)
        for j, crypto in enumerate(cryptos):
            if predictions_dict[crypto] is not None:
                num_coins = (initial_capital * allocations[j]) / crypto_prices[crypto].iloc[0]
                predicted_holdings[crypto] = predictions_dict[crypto] * num_coins
        predicted_total_holdings = predicted_holdings.sum(axis=1)

        # Define file paths
        portfolio_image = f'portfolio_{portfolio_index}_performance.png'
        individual_image = f'portfolio_{portfolio_index}_holdings.png'
        correlation_matrix_image = f'portfolio_{portfolio_index}_correlation.png'
        interactive_portfolio = f'interactive_portfolio_{portfolio_index}.html'
        distribution_image = f'portfolio_{portfolio_index}_distribution.png'
        interactive_dist = f'Portfolio_{portfolio_index}_distribution_interactive.html'

        # Save visualizations and check if PNGs were successfully created
        save_portfolio_visualizations(
            portfolio_index, total_holdings, detailed_portfolio, correlation_matrix, cryptos,
            os.path.join(STATIC_DIR, interactive_portfolio),
            os.path.join(STATIC_DIR, correlation_matrix_image),
            os.path.join(STATIC_DIR, interactive_dist)
        )

        # Check if PNG files exist; if not, set to empty strings
        individual_image_path = os.path.join(STATIC_DIR, individual_image)
        correlation_matrix_image_path = os.path.join(STATIC_DIR, correlation_matrix_image)

        individual_image_final = individual_image if os.path.exists(individual_image_path) else ''
        correlation_matrix_image_final = correlation_matrix_image if os.path.exists(
            correlation_matrix_image_path) else ''


        update_task_status(task_id, {
            'progress': 80,
            'message': 'Running Monte Carlo simulations...'
        })

        returns = total_holdings.pct_change().dropna()
        mc_paths, mc_percentiles = monte_carlo_simulation(
            returns,
            final_value,
            lstm_predictions=predicted_total_holdings.values if not predicted_total_holdings.empty else None
        )
        mc_chart = None
        if mc_paths is not None and mc_percentiles is not None:
            mc_chart = create_monte_carlo_chart(total_holdings, mc_paths, mc_percentiles, end_date, portfolio_index)

        model_prediction_chart = create_model_prediction_chart(crypto_prices, predictions_dict, end_date,
                                                               portfolio_index, ml_model)

        # Define file paths
        portfolio_image = f'portfolio_{portfolio_index}_performance.png'
        individual_image = f'portfolio_{portfolio_index}_holdings.png'
        correlation_matrix_image = f'portfolio_{portfolio_index}_correlation.png'
        interactive_portfolio = f'interactive_portfolio_{portfolio_index}.html'
        distribution_image = f'portfolio_{portfolio_index}_distribution.png'
        interactive_dist = f'Portfolio_{portfolio_index}_distribution_interactive.html'

        # Save visualizations and check if PNGs were successfully created
        save_portfolio_visualizations(
            portfolio_index, total_holdings, detailed_portfolio, correlation_matrix, cryptos,
            os.path.join(STATIC_DIR, interactive_portfolio),
            os.path.join(STATIC_DIR, correlation_matrix_image),
            os.path.join(STATIC_DIR, interactive_dist)
        )

        # Check if PNG files exist; if not, set to empty strings
        individual_image_path = os.path.join(STATIC_DIR, individual_image)
        correlation_matrix_image_path = os.path.join(STATIC_DIR, correlation_matrix_image)

        individual_image_final = individual_image if os.path.exists(individual_image_path) else ''
        correlation_matrix_image_final = correlation_matrix_image if os.path.exists(
            correlation_matrix_image_path) else ''

        # Populate portfolio_data
        portfolio_data = {
            'total_holdings': {str(k): float(v) for k, v in total_holdings.to_dict().items()},
            'initial_capital': float(initial_capital),
            'final_value': float(final_value),
            'performance': float(performance),
            'financial_metrics': {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()},
            'portfolio_image': portfolio_image,  # Placeholder, not currently used
            'individual_image': individual_image_final,  # Set to PNG path if exists, else empty
            'correlation_matrix_image': correlation_matrix_image_final,  # Set to PNG path if exists, else empty
            'interactive_portfolio': interactive_portfolio,
            'distribution_image': distribution_image,  # Placeholder, not currently used
            'interactive_distribution': interactive_dist,
            'monte_carlo_chart': mc_chart,
            'prediction_chart': model_prediction_chart,
            'ml_model': ml_model,
            'individual_interactive': f'individual_holdings_{portfolio_index}.html',
            'correlation_matrix_interactive': f'correlation_matrix_{portfolio_index}.html',
        }

        save_simulation(f"portfolio_{portfolio_index}", portfolio_data)
        update_task_status(task_id, {
            'status': 'COMPLETED',
            'progress': 100,
            'message': 'Analysis completed successfully',
            'result': portfolio_data
        })

        return {'status': 'success', 'portfolio_index': portfolio_index}

    except Exception as e:
        error_msg = f"Error processing portfolio {portfolio_index + 1}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        update_task_status(task_id, {
            'status': 'FAILED',
            'progress': 100,
            'message': f'Error: {str(e)}',
            'error': error_msg
        })
        return {'status': 'error', 'error': str(e), 'portfolio_index': portfolio_index}

@celery.task(bind=True, name='process_dca_task')
def process_dca_task(self, cryptos, allocations, start_date, capital, frequency, portfolio_index, task_id):
    logger.info(f"Processing DCA task: {task_id} for cryptos {cryptos}")
    try:
        update_task_status(task_id, {
            'status': 'PROCESSING',
            'progress': 10,
            'message': 'Starting DCA simulation...'
        })

        # Simulate DCA strategy
        result = simulate_dca_strategy(cryptos, allocations, start_date, capital, frequency)
        if result is None:
            raise ValueError("DCA simulation failed")

        # Save the result in Redis for later retrieval in /results/<batch_id>
        save_simulation(f"dca_{task_id}", result)

        update_task_status(task_id, {
            'status': 'COMPLETED',
            'progress': 100,
            'message': 'DCA simulation completed',
            'result': result
        })
        return {'status': 'success'}
    except Exception as e:
        error_msg = f"Error in DCA simulation: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        update_task_status(task_id, {
            'status': 'FAILED',
            'progress': 100,
            'message': f'Error: {str(e)}',
            'error': error_msg
        })
        return {'status': 'error', 'error': str(e)}

@app.route('/dca_simulate', methods=['POST'])
def dca_simulate():
    try:
        allocations = request.form.to_dict()
        logger.info(f"User submitted DCA allocations: {allocations}")
        start_date = request.form.get('start_date')
        capital = float(request.form.get('capital', 10000))
        frequency = request.form.get('frequency', 'weekly')
        cryptos = request.form.getlist('cryptos[]')
        allocations = {crypto: float(request.form.get(f'allocation_{crypto}', 0)) / 100 for crypto in cryptos}

        if not all([start_date, capital >= 100, cryptos, sum(allocations.values()) == 1.0]):
            logger.error("Validation failed in backend")
            return render_template('error.html', error="Invalid form data"), 400

        task_id = str(uuid.uuid4())
        logger.info(f"Queuing DCA task with task_id: {task_id}")
        update_task_status(task_id, {
            'portfolio_index': 0,  # DCA uses a single task, so index is 0
            'status': 'QUEUED',
            'progress': 0,
            'message': 'DCA simulation queued',
            'start_date': start_date,
            'capital': f"{capital:,.2f}",
            'frequency': frequency,
            'cryptos': cryptos,
            'allocations': allocations
        })

        process_dca_task.delay(cryptos, allocations, start_date, capital, frequency, 0, task_id)
        return redirect(url_for('status', batch_id=task_id))
    except Exception as e:
        logger.error(f"Error initiating DCA simulation: {str(e)}")
        return render_template('error.html', error=str(e)), 500

def create_model_prediction_chart(crypto_prices, predictions_dict, end_date, portfolio_index, model_name='LSTM'):
    try:
        last_date = pd.to_datetime(list(crypto_prices.index)[-1])
        future_dates = [last_date + timedelta(days=i) for i in range(252 + 1)][1:]
        fig = go.Figure()
        for crypto in crypto_prices.columns:
            fig.add_trace(go.Scatter(
                x=crypto_prices.index,
                y=crypto_prices[crypto],
                mode='lines',
                name=f'{crypto} Historical',
                line=dict(width=2)
            ))
        for crypto, preds in predictions_dict.items():
            if preds is not None:
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=preds,
                    mode='lines',
                    name=f'{crypto} Predicted',
                    line=dict(dash='dash', width=2)
                ))
        fig.update_layout(
            title=f'Portfolio {portfolio_index + 1} {model_name} Price Predictions (252-day Projection)',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            height=600,
            width=1200,
            showlegend=True,
            template='plotly_white'
        )
        chart_path = os.path.join(STATIC_DIR, f'{model_name.lower()}_prediction_{portfolio_index}.html')
        fig.write_html(chart_path)
        return f'{model_name.lower()}_prediction_{portfolio_index}.html'
    except Exception as e:
        logger.error(f"Error creating {model_name} chart: {str(e)}")
        return None

def update_task_status(task_id, update_data):
    current_data = get_task_status(task_id) or {}
    defaults = {
        'status': 'UNKNOWN',
        'progress': 0,
        'message': 'Task initialized',
        'portfolio_index': 0,
        'cryptos': [],
        'allocations': {},
        'start_date': 'N/A',
        'capital': 'N/A',
        'frequency': 'N/A'
    }
    current_data = {**defaults, **current_data}
    updated_data = {**current_data, **update_data}
    r.set(f"task:{task_id}", json.dumps(updated_data))

@app.route('/task_status/<task_id>')
def task_status(task_id):
    status = r.get(f"task:{task_id}")
    print(f"Fetching status for {task_id}: {status}")  # Debug
    if not status:
        return jsonify({'status': 'NOT_FOUND', 'message': 'Task not found'}), 404
    return jsonify(json.loads(status))

def get_task_status(task_id):
    try:
        data = r.get(f"task:{task_id}")
        return json.loads(data) if data else None
    except Exception as e:
        logger.error(f"Error reading task status from Redis: {str(e)}")
        return None

def save_simulation(portfolio_id, data):
    try:
        if 'total_holdings' in data and isinstance(data['total_holdings'], dict):
            data['total_holdings'] = {str(k): v for k, v in data['total_holdings'].items()}
        r.set(f"portfolio:{portfolio_id}", json.dumps(data))
        logger.info(f"Successfully saved simulation data for portfolio {portfolio_id} in Redis")
    except Exception as e:
        logger.error(f"Error saving simulation in Redis: {str(e)}")
        raise

def load_simulations():
    simulations = []
    try:
        for key in r.scan_iter("portfolio:*"):
            data = r.get(key)
            if data:
                sim_data = json.loads(data)
                if 'total_holdings' in sim_data:
                    sim_data['total_holdings'] = pd.Series(sim_data['total_holdings'])
                simulations.append(sim_data)
    except Exception as e:
        logger.error(f"Error loading simulations from Redis: {str(e)}")
    return simulations

def fetch_market_cap(crypto):
    try:
        api_url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto.lower()}&vs_currencies=usd&include_market_cap=true"
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        return data.get(crypto.lower(), {}).get('usd_market_cap')
    except Exception as e:
        logger.warning(f"Failed to fetch market cap for {crypto}: {str(e)}")
        return None

def format_portfolio_for_template(portfolio_data):
    try:
        return {
            'initial_capital': f"{float(portfolio_data['initial_capital']):,.2f}",
            'final_value': f"{float(portfolio_data['final_value']):,.2f}",
            'performance': f"{float(portfolio_data['performance']):.2f}",
            'financial_metrics': {
                k: round(v, 4) if isinstance(v, (int, float)) else 0.0
                for k, v in portfolio_data.get('financial_metrics', {}).items()
            },
            'interactive_portfolio': portfolio_data.get('interactive_portfolio', ''),
            'distribution_image': portfolio_data.get('distribution_image', ''),
            'interactive_distribution': portfolio_data.get('interactive_distribution', ''),
            'monte_carlo_chart': portfolio_data.get('monte_carlo_chart', ''),
            'prediction_chart': portfolio_data.get('prediction_chart', '')
        }
    except Exception as e:
        logger.error(f"Error formatting portfolio data: {str(e)}")
        return None

def get_historical_data(cryptos, allocations, start_date, end_date, initial_capital):
    try:
        portfolio_data = []
        crypto_prices = pd.DataFrame()
        start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
        end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
        start_timestamp = int(start_datetime.timestamp())
        end_timestamp = int(end_datetime.timestamp())
        start_cut = start_datetime
        logger.info(f"start_cut_init: {start_cut}")

        for i, crypto in enumerate(cryptos):
            kucoin_symbol = f"{crypto}-USDT"
            start_cut_kucoin = start_cut
            start_cut_yf = start_cut
            price_data_kucoin = None
            price_data_yfinance = None

            try:
                klines = kucoin_client.get_kline(symbol=kucoin_symbol, kline_type='1day', startAt=start_timestamp,
                                                 endAt=end_timestamp)
                if klines and len(klines) > 0:
                    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'amount'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
                    df.set_index('timestamp', inplace=True)
                    df = df.sort_index()
                    price_data_kucoin = df['close'].astype(float)
                    logger.info(
                        f"KuCoin data for {crypto} starts at {price_data_kucoin.index[0]} and ends at {price_data_kucoin.index[-1]}")
                    if start_cut_kucoin <= price_data_kucoin.index[0]:
                        start_cut_kucoin = price_data_kucoin.index[0]
                    logger.info(f"start_cut_kucoin: {start_cut_kucoin}")
                else:
                    logger.warning(f"No KuCoin data for {kucoin_symbol}")
            except Exception as e:
                logger.error(f"Error fetching KuCoin data for {kucoin_symbol}: {str(e)}")

            try:
                ticker = yf.Ticker(f"{crypto}-USD")
                yf_data = ticker.history(start=start_date, end=end_date, interval='1d')
                if not yf_data.empty:
                    price_data_yfinance = yf_data['Close']
                    price_data_yfinance.index = pd.to_datetime(price_data_yfinance.index).tz_localize(None)
                    logger.info(
                        f"yFinance data for {crypto} starts at {price_data_yfinance.index[0]} and ends at {price_data_yfinance.index[-1]}")
                    if start_cut_yf <= price_data_yfinance.index[0]:
                        start_cut_yf = price_data_yfinance.index[0]
                    logger.info(f"start_cut_yfi: {start_cut_yf}")
                else:
                    ticker = yf.Ticker(crypto)
                    yf_data = ticker.history(start=start_date, end=end_date, interval='1d')
                    if not yf_data.empty:
                        price_data_yfinance = yf_data['Close']
                        price_data_yfinance.index = pd.to_datetime(price_data_yfinance.index).tz_localize(None)
                        logger.info(
                            f"yFinance data for {crypto} (alt ticker) starts at {price_data_yfinance.index[0]} and ends at {price_data_yfinance.index[-1]}")
                    else:
                        logger.warning(f"No yFinance data for {crypto}")
            except Exception as e:
                logger.error(f"Error fetching yFinance data for {crypto}: {str(e)}")

            start_cut = min(start_cut_kucoin, start_cut_yf)
            price_data_list = [data for data in [price_data_kucoin, price_data_yfinance] if data is not None]
            if not price_data_list:
                logger.error(f"No data available for {crypto} from any source")
                continue

            combined_data = pd.concat(price_data_list).sort_index()
            price_data = combined_data.groupby(combined_data.index).mean()
            full_index = pd.date_range(start=start_date, end=end_date, freq='1D')
            price_data = price_data.reindex(full_index, method=None)

            price_data_non_na = price_data.dropna()
            if price_data_non_na.empty:
                logger.error(f"No valid price data for {crypto} after reindexing")
                continue
            earliest_price_date = price_data_non_na.index[0]
            logger.info(f"{crypto} earliest price date: {earliest_price_date}")

            price_data = price_data[price_data.index >= max(start_cut, earliest_price_date)]
            price_data = price_data.ffill()
            logger.info(
                f"{crypto} price_data after trimming: shape={price_data.shape}, first={price_data.index[0]}, last={price_data.index[-1]}")

            initial_investment = initial_capital * allocations[i]
            first_valid_price = price_data.iloc[0]
            num_coins = initial_investment / float(first_valid_price)
            holdings = pd.DataFrame({
                'Date': price_data.index,
                'Crypto': crypto,
                'Holdings': (price_data * num_coins),
                'Initial_Investment': initial_investment
            })
            logger.info(
                f"{crypto} Holdings: first={holdings['Holdings'].iloc[0]}, last={holdings['Holdings'].iloc[-1]}, NaN count={holdings['Holdings'].isna().sum()}")
            portfolio_data.append(holdings)
            crypto_prices[crypto] = price_data

        if not portfolio_data:
            raise ValueError("No data could be retrieved for any cryptocurrency")

        filtered_data = []
        for i, df in enumerate(portfolio_data):
            df = df[df['Date'] >= start_cut]
            logger.info(
                f"Portfolio {i} shape: {df.shape}, columns: {list(df.columns)}, first date: {df['Date'].min()}, last date: {df['Date'].max()}")
            filtered_data.append(df)

        combined_portfolio = pd.concat(filtered_data, axis=0, ignore_index=True)
        combined_portfolio['Date'] = pd.to_datetime(combined_portfolio['Date'])
        combined_portfolio = combined_portfolio.sort_values(['Date', 'Crypto'])

        logger.info(f"Combined portfolio before grouping:\n{combined_portfolio.tail(12)}")
        total_holdings = combined_portfolio.groupby('Date')['Holdings'].sum().reindex(full_index, method='ffill')
        logger.info(f"Total holdings last 5 values:\n{total_holdings.tail()}")

        crypto_prices = crypto_prices[crypto_prices.index >= start_cut]
        correlation_matrix = crypto_prices.corr()
        returns = total_holdings.pct_change().dropna()
        metrics = calculate_financial_metrics(crypto_prices, returns)

        return total_holdings, combined_portfolio, correlation_matrix, metrics, crypto_prices, None

    except Exception as e:
        logger.error(f"Fatal error in data retrieval: {str(e)}\n{traceback.format_exc()}")
        return None, None, None, None, None, None

@app.route("/")
def index():
    try:
        all_simulations = load_simulations()
        return render_template("index.html", all_simulations=all_simulations)
    except Exception as e:
        logger.error(f"Error rendering index: {str(e)}")
        return render_template("index.html", error=str(e))

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        allocations = request.form
        alloc_str = ", ".join(f"{coin}={pct}%" for coin, pct in allocations.items())
        logger.info(f"User submitted crypto allocations: {alloc_str}")
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        portfolio_count = int(request.form.get('portfolio_count', 0))
        ml_model = request.form.get('ml_model', 'LSTM')
        logger.info(f"Processing {portfolio_count} portfolios from {start_date} to {end_date} with model {ml_model}")

        batch_id = str(uuid.uuid4())
        tasks = []

        for i in range(portfolio_count):
            try:
                cryptos = request.form.getlist(f'crypto_{i}[]')
                allocations = [float(request.form.get(f'portfolio_{i}_allocation_{crypto}', 0)) / 100 for crypto in
                               cryptos]
                initial_capital = float(request.form.get(f'initial_capital_{i}', request.form.get('initial_capital')))

                task_id = f"{batch_id}_{i}"
                update_task_status(task_id, {
                    'portfolio_index': i,
                    'status': 'QUEUED',
                    'progress': 0,
                    'message': 'Job queued, waiting to start',
                    'start_date': start_date,
                    'end_date': end_date,
                    'cryptos': cryptos,
                    'allocations': allocations,
                    'initial_capital': f"{initial_capital:,.2f}"
                })

                process_portfolio_task.delay(cryptos, allocations, start_date, end_date, initial_capital, i, task_id,
                                             ml_model)
                logger.info(f"Task {task_id} queued for portfolio {i} with cryptos {cryptos}")
                tasks.append(task_id)
            except Exception as e:
                error_msg = f"Error queuing portfolio {i + 1}: {str(e)}"
                logger.error(error_msg)

        batch_data = {
            'batch_id': batch_id,
            'tasks': tasks,
            'start_date': start_date,
            'end_date': end_date,
            'portfolio_count': portfolio_count,
            'ml_model': ml_model,
            'created_at': datetime.now().isoformat()
        }

        with open(os.path.join(TASK_DIR, f"{batch_id}_batch.json"), 'w') as f:
            json.dump(batch_data, f)

        return redirect(url_for('status', batch_id=batch_id))

    except Exception as e:
        error_msg = f"Error initiating simulation: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return render_template('error.html', error=error_msg)

@app.route('/status/<batch_id>')
def status(batch_id):
    try:
        batch_file = os.path.join(TASK_DIR, f"{batch_id}_batch.json")
        is_portfolio_simulation = os.path.exists(batch_file)
        logger.info(f"Checking status for batch_id: {batch_id}, is_portfolio: {is_portfolio_simulation}")

        if is_portfolio_simulation:
            try:
                with open(batch_file, 'r') as f:
                    batch_data = json.load(f)
                task_ids = batch_data.get('tasks', [])
                total_tasks = len(task_ids)
                logger.info(f"Loaded batch file with task_ids: {task_ids}")
            except Exception as e:
                logger.error(f"Failed to load batch file {batch_file}: {str(e)}")
                task_ids = []
                total_tasks = 0
        else:
            task_ids = [batch_id]
            total_tasks = 1
            logger.info(f"DCA simulation, task_id: {batch_id}")

        task_statuses = []
        for task_id in task_ids:
            status = get_task_status(task_id)
            logger.info(f"Task {task_id} status: {status}")
            if status:
                status.setdefault('portfolio_index', 0)
                status.setdefault('status', 'UNKNOWN')
                status.setdefault('progress', 0)
                status.setdefault('message', 'No message')
                status.setdefault('start_date', 'N/A')
                status.setdefault('capital', 'N/A')
                status.setdefault('frequency', 'N/A')
                status.setdefault('cryptos', [])
                status.setdefault('allocations', {})
                task_statuses.append(status)

        logger.info(f"Task statuses: {task_statuses}")
        if not task_statuses:
            logger.warning(f"No task statuses found for batch_id: {batch_id}")
            return render_template('status.html', batch_id=batch_id, task_statuses=[],
                                   overall_progress=0, completed_tasks=0, failed_tasks=0,
                                   total_tasks=0, all_completed=False)

        completed_tasks = sum(1 for task in task_statuses if task.get('status') == 'COMPLETED')
        failed_tasks = sum(1 for task in task_statuses if task.get('status') == 'FAILED')
        overall_progress = int(
            (sum(task.get('progress', 0) for task in task_statuses) / total_tasks) if total_tasks > 0 else 0)
        all_completed = (completed_tasks + failed_tasks) == total_tasks

        logger.info(f"Rendering status: completed={completed_tasks}, failed={failed_tasks}, progress={overall_progress}")
        return render_template(
            'status.html',
            batch_id=batch_id,
            task_statuses=task_statuses,
            overall_progress=overall_progress,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            total_tasks=total_tasks,
            all_completed=all_completed
        )

    except Exception as e:
        error_msg = f"Error checking status: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return render_template('status.html', batch_id=batch_id, task_statuses=[],
                               overall_progress=0, completed_tasks=0, failed_tasks=0,
                               total_tasks=0, all_completed=False), 500

@app.route('/results/<batch_id>')
def results(batch_id):
    try:
        # Check if this is a portfolio simulation (has a batch file)
        batch_file = os.path.join(TASK_DIR, f"{batch_id}_batch.json")
        is_portfolio_simulation = os.path.exists(batch_file)

        if is_portfolio_simulation:
            # Portfolio simulation case
            with open(batch_file, 'r') as f:
                batch_data = json.load(f)
            task_ids = batch_data['tasks']
            total_tasks = len(task_ids)
            start_date = batch_data['start_date']
            end_date = batch_data['end_date']
            ml_model = batch_data.get('ml_model', 'LSTM')
        else:
            # DCA simulation case
            status = get_task_status(batch_id)
            if not status or 'result' not in status:
                return render_template('error.html', error="DCA simulation result not found")
            result = status['result']
            start_date = result['start_date']
            end_date = result['end_date']
            ml_model = 'LSTM'  # Default for DCA
            task_ids = [batch_id]
            total_tasks = 1

        # Fetch task statuses
        task_statuses = []
        for task_id in task_ids:
            status = get_task_status(task_id)
            if status:
                task_statuses.append(status)

        completed_tasks = sum(1 for task in task_statuses if task.get('status') == 'COMPLETED')
        failed_tasks = sum(1 for task in task_statuses if task.get('status') == 'FAILED')

        if completed_tasks + failed_tasks < total_tasks:
            return redirect(url_for('status', batch_id=batch_id))

        portfolios = []
        errors = []

        for task in task_statuses:
            if task.get('status') == 'COMPLETED' and task.get('result'):
                portfolio_data = task.get('result')
                if is_portfolio_simulation:
                    formatted_portfolio = format_portfolio_for_template(portfolio_data)
                    if formatted_portfolio:
                        portfolios.append(portfolio_data)
                    else:
                        logger.warning(f"Portfolio {task.get('portfolio_index', '?')} formatted as None")
                        errors.append(f"Formatting failed for portfolio {task.get('portfolio_index', '?')}")
                else:
                    # DCA case: Format result for dca_results.html
                    formatted_result = {
                        'initial_capital': f"{portfolio_data['initial_capital']:,.2f}",
                        'final_value': f"{portfolio_data['final_value']:,.2f}",
                        'performance': f"{portfolio_data['performance']:.2f}",
                        'financial_metrics': portfolio_data['financial_metrics'],
                        'interactive_portfolio': portfolio_data['interactive_portfolio'],
                        'correlation_matrix_image': portfolio_data['correlation_matrix_image'],
                        'interactive_distribution': portfolio_data['interactive_distribution'],
                        'monte_carlo_chart': portfolio_data['monte_carlo_chart'],
                        'cryptos': portfolio_data['cryptos'],
                        'allocations': [f"{alloc * 100:.0f}%" for alloc in portfolio_data['allocations']],
                        'frequency': portfolio_data['frequency'],
                        'investment_dates': portfolio_data['investment_dates'],
                        'start_date': portfolio_data['start_date'],
                        'end_date': portfolio_data['end_date'],
                        'total_holdings': {k: v for k, v in portfolio_data['total_holdings'].items()}
                    }
                    return render_template('dca_results.html', result=formatted_result)
            elif task.get('status') == 'FAILED':
                errors.append(task.get('error', f"Unknown error in portfolio {task.get('portfolio_index', '?')}"))

        if not portfolios and errors:
            logger.info(f"No portfolios completed for batch {batch_id}. Errors: {errors}")

        portfolio_correlation = None
        combined_chart = None
        combined_interactive = None
        heatmap_filename = None

        if len(portfolios) > 1:
            try:
                correlation_data = pd.DataFrame({
                    f'Portfolio {i + 1}': pd.Series(p['total_holdings'])
                    for i, p in enumerate(portfolios)
                })
                portfolio_correlation = correlation_data.corr()
                heatmap_filename = save_correlation_heatmap(portfolio_correlation)
                combined_chart, combined_interactive = save_combined_portfolio_chart(portfolios,
                                                                                     batch_data['start_date'],
                                                                                     batch_data['end_date'])
            except Exception as e:
                logger.error(f"Error creating correlation matrix or combined chart: {str(e)}")
                errors.append(f"Visualization error: {str(e)}")

        formatted_portfolios = [format_portfolio_for_template(p) for p in portfolios]

        return render_template(
            'results.html',
            start_date=start_date,
            end_date=end_date,
            portfolios=formatted_portfolios,
            portfolio_correlation=portfolio_correlation,
            combined_chart=combined_chart,
            combined_interactive=combined_interactive,
            heatmap_filename=heatmap_filename,
            error="\n".join(errors) if errors else None,
            ml_model=ml_model
        )

    except Exception as e:
        error_msg = f"Error displaying results: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return render_template('error.html', error=error_msg)


@app.route('/donate')
def donate():
    return render_template('donate.html', current_year=datetime.now().strftime('%Y'))

def simulate_dca_strategy(cryptos, allocations, start_date, capital, frequency='weekly', end_date=None):
    try:
        logger.info(f"Simulating DCA for {cryptos} starting {start_date} with ${capital}, frequency: {frequency}")

        # Default end_date to today if not provided
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        # Fetch historical data
        total_holdings, detailed_portfolio, correlation_matrix, metrics, crypto_prices, _ = get_historical_data(
            cryptos, allocations, start_date, end_date, capital
        )
        if total_holdings is None:
            raise ValueError("Failed to fetch historical data for DCA simulation")

        # Define frequency in days
        freq_days = {'daily': 1, 'weekly': 7, 'monthly': 30}
        investment_interval = freq_days.get(frequency, 7)  # Default to weekly
        num_investments = len(total_holdings) // investment_interval
        if num_investments < 1:
            raise ValueError(f"Not enough data for {frequency} DCA simulation")

        # Calculate per-investment amount
        investment_per_period = capital / num_investments
        dca_holdings = pd.DataFrame(index=total_holdings.index, columns=cryptos)
        dca_totals = pd.Series(0, index=total_holdings.index)
        investment_dates = []

        for crypto, alloc in zip(cryptos, allocations):
            num_coins = 0
            prices = crypto_prices[crypto].dropna()
            for i, date in enumerate(prices.index):
                if i % investment_interval == 0 and i < num_investments * investment_interval:
                    investment_amount = investment_per_period * alloc
                    price = prices.iloc[i]
                    coins_bought = investment_amount / price
                    num_coins += coins_bought
                    if crypto == cryptos[0]:  # Track investment dates only once
                        investment_dates.append(date)
                dca_holdings[crypto].iloc[i] = num_coins * prices.iloc[i] if i < len(prices) else \
                    dca_holdings[crypto].iloc[i - 1]
            dca_holdings[crypto] = dca_holdings[crypto].fillna(method='ffill')

        dca_totals = dca_holdings.sum(axis=1)
        final_value = dca_totals.iloc[-1]
        performance = ((final_value - capital) / capital) * 100

        # Generate visualizations
        interactive_portfolio = f'dca_interactive_portfolio_{uuid.uuid4()}.html'
        correlation_matrix_image = f'dca_correlation_{uuid.uuid4()}.html'
        interactive_dist = f'dca_distribution_interactive_{uuid.uuid4()}.html'

        save_portfolio_visualizations(
            0, dca_totals, detailed_portfolio, correlation_matrix, cryptos,
            os.path.join(STATIC_DIR, interactive_portfolio),
            os.path.join(STATIC_DIR, correlation_matrix_image),
            os.path.join(STATIC_DIR, interactive_dist)
        )

        # Financial metrics
        returns = dca_totals.pct_change().dropna()
        metrics = calculate_financial_metrics(crypto_prices, returns)

        # Monte Carlo simulation
        mc_paths, mc_percentiles = monte_carlo_simulation(returns, final_value)
        mc_chart = create_monte_carlo_chart(dca_totals, mc_paths, mc_percentiles, end_date,
                                            0) if mc_paths is not None else None

        result = {
            'total_holdings': dca_totals.to_dict(),
            'initial_capital': float(capital),
            'final_value': float(final_value),
            'performance': float(performance),
            'financial_metrics': metrics,
            'interactive_portfolio': interactive_portfolio,
            'correlation_matrix_image': correlation_matrix_image,
            'interactive_distribution': interactive_dist,
            'monte_carlo_chart': mc_chart,
            'cryptos': cryptos,
            'allocations': allocations,
            'frequency': frequency,
            'investment_dates': [date.strftime('%Y-%m-%d') for date in investment_dates],
            'start_date': start_date,
            'end_date': end_date
        }

        return result

    except Exception as e:
        logger.error(f"Error in DCA simulation: {str(e)}\n{traceback.format_exc()}")
        return None

@app.route('/dca', methods=['POST'])
def dca():
    try:
        start_date = request.form.get('start_date')
        capital = float(request.form.get('capital'))
        frequency = request.form.get('frequency', 'weekly')
        cryptos = request.form.getlist('cryptos[]')
        allocations = [float(request.form.get(f'allocation_{crypto}', 0)) / 100 for crypto in cryptos]

        if not cryptos or abs(sum(allocations) - 1.0) > 0.01:
            raise ValueError("Invalid cryptocurrency selections or allocations (must sum to 100%)")

        result = simulate_dca_strategy(cryptos, allocations, start_date, capital, frequency)
        if result is None:
            raise ValueError("DCA simulation failed")

        formatted_result = {
            'initial_capital': f"{result['initial_capital']:,.2f}",
            'final_value': f"{result['final_value']:,.2f}",
            'performance': f"{result['performance']:.2f}",
            'financial_metrics': result['financial_metrics'],
            'interactive_portfolio': result['interactive_portfolio'],
            'correlation_matrix_image': result['correlation_matrix_image'],
            'interactive_distribution': result['interactive_distribution'],
            'monte_carlo_chart': result['monte_carlo_chart'],
            'cryptos': result['cryptos'],
            'allocations': [f"{alloc * 100:.0f}%" for alloc in result['allocations']],
            'frequency': result['frequency'],
            'investment_dates': result['investment_dates'],
            'start_date': result['start_date'],
            'end_date': result['end_date'],
            'total_holdings': {k: v for k, v in result['total_holdings'].items()}
        }

        return render_template('dca_results.html', result=formatted_result)

    except Exception as e:
        logger.error(f"Error in DCA route: {str(e)}\n{traceback.format_exc()}")
        return render_template('error.html', error=str(e))

def simulate_mean_reversion_trading(spread, position_size, beta, prices1, prices2, initial_capital, transaction_cost=0.001):
    """
    Simulate an enhanced mean-reversion pair trading strategy with dynamic thresholds and risk management.
    Buy crypto1 and short beta*crypto2 when spread is below dynamic threshold,
    sell crypto1 and cover beta*crypto2 when above threshold.
    Includes transaction costs, trailing stop, and volatility-adjusted sizing.
    """
    try:
        # Calculate rolling mean and std for dynamic thresholds
        window = 30  # 30-day rolling window
        spread_mean = spread.rolling(window=window).mean()
        spread_std = spread.rolling(window=window).std()
        z_score = (spread - spread_mean) / spread_std

        # Generate trading signals: 1 for buy, -1 for sell, 0 for hold
        signals = pd.DataFrame(index=spread.index)
        signals['z_score'] = z_score
        signals['signal'] = 0
        signals.loc[z_score < -1.5, 'signal'] = 1  # Buy when z-score < -1.5
        signals.loc[z_score > 1.5, 'signal'] = -1  # Sell when z-score > 1.5

        # Initialize trading variables
        position = 0  # 0 = no position, 1 = long, -1 = short
        units_crypto1 = 0  # Units of crypto1 held
        units_crypto2 = 0  # Units of crypto2 held (short)
        cash = initial_capital
        total_pnl = 0
        pnl_series = pd.Series(0, index=spread.index)
        entry_spread = 0
        trailing_stop = 0
        min_holding_period = 5  # Minimum 5 days to reduce transaction costs
        days_held = 0

        for i in range(window, len(signals)):
            current_price1 = prices1.iloc[i]
            current_price2 = prices2.iloc[i]
            current_spread = spread.iloc[i]
            prev_signal = signals['signal'].iloc[i-1]
            current_signal = signals['signal'].iloc[i]

            # Volatility-adjusted position size
            spread_volatility = spread_std.iloc[i] * current_price1
            trade_value = min(cash * 0.05, position_size / (1 + spread_volatility / current_price1))
            if trade_value <= 0:
                continue

            units_to_trade1 = trade_value / current_price1
            units_to_trade2 = units_to_trade1 * beta

            # Update trailing stop if in a position
            if position == 1:
                trailing_stop = max(trailing_stop, current_spread * (1 - 0.01))  # 1% trailing stop
            elif position == -1:
                trailing_stop = min(trailing_stop, current_spread * (1 + 0.01))

            # Enter or exit positions
            if prev_signal == 0 and current_signal == 1:  # Enter long
                cost = trade_value * (1 + transaction_cost)
                if cash >= cost:
                    units_crypto1 += units_to_trade1
                    units_crypto2 -= units_to_trade2
                    cash -= cost
                    entry_spread = current_spread
                    trailing_stop = entry_spread
                    position = 1
                    days_held = 0
                    logger.info(f"Buy at spread {entry_spread}, units1={units_crypto1}, units2={units_crypto2}, cash={cash}")

            elif prev_signal == 0 and current_signal == -1:  # Enter short
                cost = trade_value * (1 + transaction_cost)
                if cash >= cost:
                    units_crypto1 -= units_to_trade1
                    units_crypto2 += units_to_trade2
                    cash -= cost
                    entry_spread = current_spread
                    trailing_stop = entry_spread
                    position = -1
                    days_held = 0
                    logger.info(f"Sell at spread {entry_spread}, units1={units_crypto1}, units2={units_crypto2}, cash={cash}")

            elif position != 0:
                days_held += 1
                portfolio_value = (units_crypto1 * current_price1 + units_crypto2 * current_price2)
                trade_pnl = portfolio_value - (trade_value if position == 1 else -trade_value)

                # Exit conditions: z-score returns to mean, trailing stop hit, or minimum holding period
                if (position == 1 and current_spread >= trailing_stop) or \
                   (position == -1 and current_spread <= trailing_stop) or \
                   (current_signal == 0 and days_held >= min_holding_period):
                    exit_value = portfolio_value
                    trade_pnl = exit_value - (trade_value if position == 1 else -trade_value)
                    trade_pnl -= abs(exit_value) * transaction_cost
                    total_pnl += trade_pnl
                    cash += exit_value - (abs(exit_value) * transaction_cost)
                    units_crypto1 = 0
                    units_crypto2 = 0
                    position = 0
                    days_held = 0
                    logger.info(f"Exit at spread {current_spread}, P&L={trade_pnl}, cash={cash}")

            # Update P&L series
            portfolio_value = (units_crypto1 * current_price1 + units_crypto2 * current_price2) + cash
            pnl_series.iloc[i] = total_pnl + (portfolio_value - initial_capital)

        return signals, pnl_series

    except Exception as e:
        logger.error(f"Error simulating mean-reversion trading: {str(e)}")
        return pd.DataFrame(), pd.Series()

def simulate_bollinger_bands_trading(spread, position_size, beta, prices1, prices2, initial_capital, transaction_cost=0.001):
    """
    Simulate a Bollinger Bands pair trading strategy.
    Buy when spread crosses below lower band, sell when above upper band.
    """
    try:
        window = 20  # 20-day window for Bollinger Bands
        spread_ma = spread.rolling(window=window).mean()
        spread_std = spread.rolling(window=window).std()
        upper_band = spread_ma + 2 * spread_std
        lower_band = spread_ma - 2 * spread_std

        signals = pd.DataFrame(index=spread.index)
        signals['spread'] = spread
        signals['signal'] = 0
        signals.loc[spread < lower_band, 'signal'] = 1  # Buy
        signals.loc[spread > upper_band, 'signal'] = -1  # Sell

        position = 0
        units_crypto1 = 0
        units_crypto2 = 0
        cash = initial_capital
        total_pnl = 0
        pnl_series = pd.Series(0, index=spread.index)

        for i in range(window, len(signals)):
            current_price1 = prices1.iloc[i]
            current_price2 = prices2.iloc[i]
            current_spread = spread.iloc[i]
            prev_signal = signals['signal'].iloc[i-1]
            current_signal = signals['signal'].iloc[i]

            trade_value = min(cash * 0.05, position_size)
            units_to_trade1 = trade_value / current_price1
            units_to_trade2 = units_to_trade1 * beta

            if prev_signal == 0 and current_signal == 1:  # Enter long
                cost = trade_value * (1 + transaction_cost)
                if cash >= cost:
                    units_crypto1 += units_to_trade1
                    units_crypto2 -= units_to_trade2
                    cash -= cost
                    position = 1

            elif prev_signal == 0 and current_signal == -1:  # Enter short
                cost = trade_value * (1 + transaction_cost)
                if cash >= cost:
                    units_crypto1 -= units_to_trade1
                    units_crypto2 += units_to_trade2
                    cash -= cost
                    position = -1

            elif position != 0 and abs(current_spread - spread_ma.iloc[i]) < spread_std.iloc[i] * 0.5:  # Exit near mean
                exit_value = (units_crypto1 * current_price1 + units_crypto2 * current_price2)
                trade_pnl = exit_value - (trade_value if position == 1 else -trade_value)
                trade_pnl -= abs(exit_value) * transaction_cost
                total_pnl += trade_pnl
                cash += exit_value - (abs(exit_value) * transaction_cost)
                units_crypto1 = 0
                units_crypto2 = 0
                position = 0

            portfolio_value = (units_crypto1 * current_price1 + units_crypto2 * current_price2) + cash
            pnl_series.iloc[i] = total_pnl + (portfolio_value - initial_capital)

        return signals, pnl_series

    except Exception as e:
        logger.error(f"Error simulating Bollinger Bands trading: {str(e)}")
        return pd.DataFrame(), pd.Series()

def simulate_correlation_trading(prices1, prices2, position_size, beta, initial_capital, transaction_cost=0.001):
    """
    Simulate a correlation-based pair trading strategy.
    Trade when correlation drops below threshold, exit when it reverts.
    """
    try:
        window = 20
        rolling_corr = prices1.rolling(window=window).corr(prices2)
        signals = pd.DataFrame(index=prices1.index)
        signals['corr'] = rolling_corr
        signals['signal'] = 0
        signals.loc[rolling_corr < 0.5, 'signal'] = 1  # Buy spread
        signals.loc[rolling_corr > 0.8, 'signal'] = -1  # Sell spread

        position = 0
        units_crypto1 = 0
        units_crypto2 = 0
        cash = initial_capital
        total_pnl = 0
        pnl_series = pd.Series(0, index=prices1.index)

        for i in range(window, len(signals)):
            current_price1 = prices1.iloc[i]
            current_price2 = prices2.iloc[i]
            prev_signal = signals['signal'].iloc[i-1]
            current_signal = signals['signal'].iloc[i]

            trade_value = min(cash * 0.05, position_size)
            units_to_trade1 = trade_value / current_price1
            units_to_trade2 = units_to_trade1 * beta

            if prev_signal == 0 and current_signal == 1:  # Enter long
                cost = trade_value * (1 + transaction_cost)
                if cash >= cost:
                    units_crypto1 += units_to_trade1
                    units_crypto2 -= units_to_trade2
                    cash -= cost
                    position = 1

            elif prev_signal == 0 and current_signal == -1:  # Enter short
                cost = trade_value * (1 + transaction_cost)
                if cash >= cost:
                    units_crypto1 -= units_to_trade1
                    units_crypto2 += units_to_trade2
                    cash -= cost
                    position = -1

            elif position != 0 and current_signal == 0:  # Exit
                exit_value = (units_crypto1 * current_price1 + units_crypto2 * current_price2)
                trade_pnl = exit_value - (trade_value if position == 1 else -trade_value)
                trade_pnl -= abs(exit_value) * transaction_cost
                total_pnl += trade_pnl
                cash += exit_value - (abs(exit_value) * transaction_cost)
                units_crypto1 = 0
                units_crypto2 = 0
                position = 0

            portfolio_value = (units_crypto1 * current_price1 + units_crypto2 * current_price2) + cash
            pnl_series.iloc[i] = total_pnl + (portfolio_value - initial_capital)

        return signals, pnl_series

    except Exception as e:
        logger.error(f"Error simulating correlation trading: {str(e)}")
        return pd.DataFrame(), pd.Series()




def create_trading_pnl_chart(pnl_series, crypto1, crypto2, chart_path):
    """
    Create an interactive Plotly chart for trading P&L.
    """
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pnl_series.index,
            y=pnl_series.cumsum(),  # Cumulative P&L
            mode='lines',
            name=f'Trading P&L ({crypto1} vs {crypto2})',
            line=dict(color='green', width=2)
        ))
        fig.update_layout(
            title=f"Trading Profit and Loss ({crypto1} vs {crypto2})",
            xaxis_title="Date",
            yaxis_title="Cumulative P&L (USD)",
            height=600,
            width=1200,
            showlegend=True,
            template='plotly_white'
        )
        fig.write_html(chart_path)
        logger.info(f"Trading P&L chart saved to {chart_path}")
    except Exception as e:
        logger.error(f"Error creating trading P&L chart: {str(e)}")


def simulate_stationary_portfolio(spread, initial_capital):
    """
    Simulate the value of a stationary portfolio based on the spread.
    Assumes the spread is mean-reverting and scales it to initial capital.
    """
    try:
        # Normalize spread to start at initial_capital
        spread_mean = spread.mean()
        spread_std = spread.std()
        normalized_spread = (spread - spread_mean) / spread_std  # Standardize spread
        portfolio_value = initial_capital + (normalized_spread * initial_capital * 0.1)  # Scale movements
        return pd.Series(portfolio_value, index=spread.index)
    except Exception as e:
        logger.error(f"Error simulating stationary portfolio: {str(e)}")
        return None

def create_stationary_portfolio_chart(portfolio_value, crypto1, crypto2, chart_path):
    """
    Create an interactive Plotly chart for the stationary portfolio's performance.
    """
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=portfolio_value.index,
            y=portfolio_value.values,
            mode='lines',
            name=f'{crypto1} - β * {crypto2}',
            line=dict(color='purple', width=2)
        ))
        fig.update_layout(
            title=f"Stationary Portfolio Performance ({crypto1} vs {crypto2})",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (USD)",
            height=600,
            width=1200,
            showlegend=True,
            template='plotly_white'
        )
        fig.write_html(chart_path)
        logger.info(f"Stationary portfolio chart saved to {chart_path}")
    except Exception as e:
        logger.error(f"Error creating stationary portfolio chart: {str(e)}")


@app.route('/cointegration', methods=['POST'])
def cointegration():
    try:
        crypto1 = request.form.get('crypto1')
        crypto2 = request.form.get('crypto2')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        initial_capital = float(request.form.get('initial_capital', 1000000))
        strategy = request.form.get('strategy', 'mean_reversion')  # Default to mean_reversion

        if not all([crypto1, crypto2, start_date, end_date]):
            raise ValueError("All fields are required.")
        if crypto1 == crypto2:
            raise ValueError("Please select two different cryptocurrencies.")

        # Fetch historical data for both cryptos
        result1 = get_historical_data([crypto1], [1.0], start_date, end_date, initial_capital)
        result2 = get_historical_data([crypto2], [1.0], start_date, end_date, initial_capital)

        if result1[0] is None or result2[0] is None:
            raise ValueError(f"Failed to retrieve data for {crypto1} or {crypto2}.")

        prices1 = result1[4][crypto1]
        prices2 = result2[4][crypto2]

        # Align data
        combined = pd.concat([prices1, prices2], axis=1).dropna()
        if len(combined) < 30:
            raise ValueError("Insufficient overlapping data for co-integration test (minimum 30 days required).")

        # Engle-Granger co-integration test
        score, p_value, critical_values = coint(combined[crypto1], combined[crypto2])
        is_cointegrated = p_value < 0.05

        result = {
            'crypto1': crypto1,
            'crypto2': crypto2,
            'start_date': start_date,
            'end_date': end_date,
            'score': round(score, 4),
            'p_value': round(p_value, 4),
            'is_cointegrated': is_cointegrated,
            'message': ("The two cryptocurrencies are co-integrated (long-term relationship exists)."
                       if is_cointegrated else "No co-integration detected between the two cryptocurrencies."),
            'stationary_portfolio_chart': None,
            'trading_pnl_chart': None
        }

        if is_cointegrated:
            # Calculate hedge ratio (beta)
            X = combined[crypto2].values.reshape(-1, 1)
            y = combined[crypto1].values
            beta = np.linalg.lstsq(X, y, rcond=None)[0][0]

            # Construct stationary portfolio
            spread = combined[crypto1] - beta * combined[crypto2]
            portfolio_value = simulate_stationary_portfolio(spread, initial_capital)

            # Simulate trading based on selected strategy
            position_size = initial_capital * 0.1
            if strategy == 'mean_reversion':
                trading_signals, pnl_series = simulate_mean_reversion_trading(spread, position_size, beta, prices1, prices2, initial_capital)
            elif strategy == 'bollinger_bands':
                trading_signals, pnl_series = simulate_bollinger_bands_trading(spread, position_size, beta, prices1, prices2, initial_capital)
            elif strategy == 'correlation':
                trading_signals, pnl_series = simulate_correlation_trading(prices1, prices2, position_size, beta, initial_capital)
            else:
                raise ValueError("Invalid strategy selected.")

            # Generate charts
            stationary_chart_filename = f"stationary_portfolio_{crypto1}_{crypto2}_{uuid.uuid4()}.html"
            stationary_chart_path = os.path.join(STATIC_DIR, stationary_chart_filename)
            create_stationary_portfolio_chart(portfolio_value, crypto1, crypto2, stationary_chart_path)

            trading_pnl_chart_filename = f"trading_pnl_{crypto1}_{crypto2}_{uuid.uuid4()}.html"
            trading_pnl_chart_path = os.path.join(STATIC_DIR, trading_pnl_chart_filename)
            create_trading_pnl_chart(pnl_series, crypto1, crypto2, trading_pnl_chart_path)

            # Update result
            result['stationary_portfolio_chart'] = stationary_chart_filename
            result['trading_pnl_chart'] = trading_pnl_chart_filename
            result['hedge_ratio'] = round(beta, 4)
            result['total_pnl'] = round(pnl_series.iloc[-1] if not pnl_series.empty else 0, 2)
            result['win_rate'] = round((trading_signals['signal'].value_counts().get(1, 0) / len(trading_signals[trading_signals['signal'] != 0])) * 100, 2) if len(trading_signals[trading_signals['signal'] != 0]) > 0 else 0
            result['message'] += (f" Stationary portfolio constructed with hedge ratio (beta) = {round(beta, 4)}. "
                                 f"Total P&L from trading ({strategy.replace('_', ' ').title()}): ${result['total_pnl']:.2f}, "
                                 f"Win Rate: {result['win_rate']}%, with position size {position_size:.1f}.")

        logger.info(f"Co-integration test completed: {crypto1} vs {crypto2}, p-value: {p_value}, cointegrated: {is_cointegrated}")
        return render_template('cointegration_results.html', result=result)

    except Exception as e:
        error_msg = f"Error in co-integration test: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return render_template('error.html', error=error_msg)


def save_correlation_heatmap(portfolio_correlation, filename='correlation_heatmap.html'):
    fig = go.Figure(data=go.Heatmap(
        z=portfolio_correlation.values,
        x=portfolio_correlation.columns,
        y=portfolio_correlation.index,
        colorscale='RdBu',
        zmid=0,
        text=portfolio_correlation.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    fig.update_layout(title='Portfolio Correlation Heatmap', height=600, width=800)
    fig.write_html(os.path.join(STATIC_DIR, filename))
    return filename

def save_combined_portfolio_chart(portfolios, start_date, end_date):
    try:
        fig = make_subplots(rows=1, cols=1, subplot_titles=("Combined Portfolios Performance"))
        for i, portfolio in enumerate(portfolios):
            total_holdings = pd.Series(portfolio['total_holdings'])
            fig.add_trace(
                go.Scatter(x=total_holdings.index, y=total_holdings.values, mode='lines', name=f'Portfolio {i + 1}'),
                row=1, col=1)
        fig.update_layout(height=600, width=1200, title_text="Combined Portfolios Performance")
        static_chart_path = os.path.join(STATIC_DIR, 'combined_chart.png')
        fig.write_image(static_chart_path, engine="kaleido")
        interactive_chart_path = os.path.join(STATIC_DIR, 'combined_interactive.html')
        fig.write_html(interactive_chart_path)
        return 'combined_chart.png', 'combined_interactive.html'
    except Exception as e:
        logger.error(f"Failed to save combined chart: {str(e)}")
        return None, None

def calculate_financial_metrics(prices, returns, risk_free_rate=0.02):
    try:
        metrics = {}
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]
        returns_array = np.array(returns.dropna()).flatten()
        metrics['Mean Return'] = returns_array.mean() * 252
        metrics['Volatility'] = returns_array.std() * np.sqrt(252)
        excess_returns = returns_array - (risk_free_rate / 252)
        metrics['Sharpe Ratio'] = (
                excess_returns.mean() / excess_returns.std() * np.sqrt(252)) if excess_returns.std() != 0 else 0
        downside_returns = returns_array[returns_array < 0]
        metrics['Sortino Ratio'] = (returns_array.mean() / downside_returns.std() * np.sqrt(252)) if len(
            downside_returns) > 0 and downside_returns.std() != 0 else 0
        cumulative_returns = (1 + returns_array).cumprod()
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        metrics['Max Drawdown'] = drawdown.min() * 100 if len(drawdown) > 0 else 0
        metrics['VaR 95%'] = np.percentile(returns_array, 5) * 100
        metrics['Return Skewness'] = stats.skew(returns_array) if len(returns_array) > 2 else 0
        metrics['Return Kurtosis'] = stats.kurtosis(returns_array) if len(returns_array) > 2 else 0
        return {k: round(v, 4) for k, v in metrics.items()}
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}\n{traceback.format_exc()}")
        return {}

def train_lstm_model(price_series, lookback=60, forecast_days=252):
    from tensorflow.keras import Sequential, Input
    from tensorflow.keras.layers import LSTM, Dropout, Dense
    from keras.src.optimizers import Adam

    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(price_series.values.reshape(-1, 1))

        if len(scaled_data) < lookback:
            logger.error(f"Insufficient data: {len(scaled_data)} points available, need at least {lookback}")
            return None

        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i - lookback:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        train_size = int(len(X) * 0.8)
        if train_size == 0:
            logger.error("Not enough data for training after splitting")
            return None

        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        model = Sequential([
            Input(shape=(lookback, 1)),
            LSTM(units=32, return_sequences=True),
            Dropout(0.2),
            LSTM(units=16, return_sequences=False),
            Dropout(0.2),
            Dense(units=16),
            Dense(units=1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test), verbose=0)

        last_sequence = scaled_data[-lookback:]
        future_predictions = []
        for i in range(forecast_days):
            last_sequence_reshaped = last_sequence.reshape((1, lookback, 1))
            next_pred = model.predict(last_sequence_reshaped, verbose=0)
            future_predictions.append(next_pred[0, 0])
            last_sequence = np.append(last_sequence[1:], next_pred)

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        future_predictions = np.clip(future_predictions, a_min=1e-6, a_max=None)

        if len(future_predictions) != forecast_days:
            logger.error(f"Generated {len(future_predictions)} predictions instead of {forecast_days}")

        return future_predictions.flatten()
    except Exception as e:
        logger.error(f"Error in train_lstm_model: {str(e)}")
        return None

def preprocess_data(price_series, lookback=60, forecast_days=252, test_split=0.2):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(price_series.values.reshape(-1, 1))
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    train_size = int(len(X) * (1 - test_split))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test, scaler, scaled_data

def train_gru_model(price_series, lookback=60, forecast_days=252):
    from tensorflow.keras import Sequential, Input
    from tensorflow.keras.layers import Dropout, Dense, GRU
    from keras.src.optimizers import Adam
    try:
        X_train, X_test, y_train, y_test, scaler, scaled_data = preprocess_data(price_series, lookback, forecast_days)
        model = Sequential([
            Input(shape=(lookback, 1)),
            GRU(units=64, return_sequences=True),
            Dropout(0.2),
            GRU(units=32, return_sequences=False),
            Dropout(0.2),
            Dense(units=16),
            Dense(units=1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test),
                  callbacks=[early_stopping], verbose=0)
        future_predictions = generate_predictions(model, scaled_data, lookback, forecast_days, scaler)
        return future_predictions
    except Exception as e:
        logger.error(f"Error training GRU model: {str(e)}")
        return None

def train_bidirectional_rnn(price_series, lookback=60, forecast_days=252, rnn_type='LSTM'):
    from tensorflow.keras import Sequential, Input
    from tensorflow.keras.layers import LSTM, Dropout, Dense, GRU
    from keras.src.optimizers import Adam
    try:
        X_train, X_test, y_train, y_test, scaler, scaled_data = preprocess_data(price_series, lookback, forecast_days)
        rnn_layer = LSTM if rnn_type.upper() == 'LSTM' else GRU
        model = Sequential([
            Input(shape=(lookback, 1)),
            Bidirectional(rnn_layer(units=64, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(rnn_layer(units=32, return_sequences=False)),
            Dropout(0.2),
            Dense(units=16),
            Dense(units=1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test),
                  callbacks=[early_stopping], verbose=0)
        future_predictions = generate_predictions(model, scaled_data, lookback, forecast_days, scaler)
        return future_predictions
    except Exception as e:
        logger.error(f"Error training Bidirectional RNN model: {str(e)}")
        return None

def train_1d_cnn(price_series, lookback=60, forecast_days=252):
    from tensorflow.keras import Sequential, Input
    from tensorflow.keras.layers import Dropout, Dense
    from keras.src.optimizers import Adam
    from keras.src.layers import Flatten
    try:
        X_train, X_test, y_train, y_test, scaler, scaled_data = preprocess_data(price_series, lookback, forecast_days)
        model = Sequential([
            Input(shape=(lookback, 1)),
            Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(units=16, activation='relu'),
            Dropout(0.2),
            Dense(units=1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test),
                  callbacks=[early_stopping], verbose=0)
        future_predictions = generate_predictions(model, scaled_data, lookback, forecast_days, scaler)
        return future_predictions
    except Exception as e:
        logger.error(f"Error training 1D CNN model: {str(e)}")
        return None

def train_cnn_gru(price_series, lookback=60, forecast_days=252):
    from tensorflow.keras import Input
    from tensorflow.keras.layers import Dropout, Dense, GRU
    from keras.src.optimizers import Adam
    try:
        X_train, X_test, y_train, y_test, scaler, scaled_data = preprocess_data(price_series, lookback, forecast_days)
        input_layer = Input(shape=(lookback, 1))
        cnn = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        cnn = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(cnn)
        gru = GRU(units=64, return_sequences=True)(cnn)
        gru = Dropout(0.2)(gru)
        gru = GRU(units=32, return_sequences=False)(gru)
        gru = Dropout(0.2)(gru)
        dense = Dense(units=16)(gru)
        output = Dense(units=1)(dense)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test),
                  callbacks=[early_stopping], verbose=0)
        future_predictions = generate_predictions(model, scaled_data, lookback, forecast_days, scaler)
        return future_predictions
    except Exception as e:
        logger.error(f"Error training CNN-GRU model: {str(e)}")
        return None

def train_transformer_model(price_series, lookback=60, forecast_days=252):
    from keras.src.optimizers import Adam
    try:
        X_train, X_test, y_train, y_test, scaler, scaled_data = preprocess_data(price_series, lookback, forecast_days)

        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
            x = LayerNormalization(epsilon=1e-6)(inputs)
            x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
            x = Dropout(dropout)(x)
            res = Add()([x, inputs])
            x = LayerNormalization(epsilon=1e-6)(res)
            x = Dense(ff_dim, activation="relu")(x)
            x = Dropout(dropout)(x)
            x = Dense(inputs.shape[-1])(x)
            return Add()([x, res])

        input_layer = Input(shape=(lookback, 1))
        x = input_layer
        for _ in range(2):
            x = transformer_encoder(x, head_size=32, num_heads=2, ff_dim=64, dropout=0.2)
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation="relu")(x)
        x = Dropout(0.2)(x)
        output = Dense(1)(x)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test),
                  callbacks=[early_stopping], verbose=0)
        future_predictions = generate_predictions(model, scaled_data, lookback, forecast_days, scaler)
        return future_predictions
    except Exception as e:
        logger.error(f"Error training Transformer model: {str(e)}")
        return None

def train_time_distributed_model(price_series, lookback=60, forecast_days=252):
    try:
        X_train, X_test, y_train, y_test, scaler, scaled_data = preprocess_data(price_series, lookback, forecast_days)
        sub_seq_len = 5
        if lookback % sub_seq_len != 0:
            new_lookback = (lookback // sub_seq_len) * sub_seq_len
            X_train = X_train[:, -new_lookback:, :]
            X_test = X_test[:, -new_lookback:, :]
            lookback = new_lookback
        X_train = X_train.reshape(X_train.shape[0], lookback // sub_seq_len, sub_seq_len, 1)
        X_test = X_test.reshape(X_test.shape[0], lookback // sub_seq_len, sub_seq_len, 1)
        input_layer = Input(shape=(lookback // sub_seq_len, sub_seq_len, 1))
        x = TimeDistributed(Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'))(input_layer)
        x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
        from keras.src.layers import Flatten
        x = TimeDistributed(Flatten())(x)
        from keras.src.layers import LSTM
        x = LSTM(64, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        output = Dense(1)(x)
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test),
                  callbacks=[early_stopping], verbose=0)
        last_sequence = scaled_data[-lookback:]
        future_predictions = []
        for _ in range(forecast_days):
            sequence_reshaped = last_sequence.reshape(1, lookback // sub_seq_len, sub_seq_len, 1)
            next_pred = model.predict(sequence_reshaped, verbose=0)
            future_predictions.append(next_pred[0, 0])
            last_sequence = np.append(last_sequence[1:], next_pred)
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        future_predictions = np.clip(future_predictions, a_min=1e-6, a_max=None)
        return future_predictions.flatten()
    except Exception as e:
        logger.error(f"Error training TimeDistributed model: {str(e)}")
        return None

def generate_predictions(model, scaled_data, lookback, forecast_days, scaler):
    last_sequence = scaled_data[-lookback:]
    future_predictions = []
    for _ in range(forecast_days):
        last_sequence_reshaped = last_sequence.reshape((1, lookback, 1))
        next_pred = model.predict(last_sequence_reshaped, verbose=0)
        future_predictions.append(next_pred[0, 0])
        last_sequence = np.append(last_sequence[1:], next_pred)
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    future_predictions = np.clip(future_predictions, a_min=1e-6, a_max=None)
    return future_predictions.flatten()

def select_model_for_prediction(price_series, ml_model, lookback=60, forecast_days=252):
    logger.info(f"Training {ml_model} model for price prediction")
    model_functions = {
        'LSTM': train_lstm_model,
        'GRU': train_gru_model,
        'BiLSTM': lambda price_series, lookback, forecast_days: train_bidirectional_rnn(price_series, lookback,
                                                                                        forecast_days, rnn_type='LSTM'),
        'BiGRU': lambda price_series, lookback, forecast_days: train_bidirectional_rnn(price_series, lookback,
                                                                                       forecast_days, rnn_type='GRU'),
        'CNN': train_1d_cnn,
        'CNN-GRU': train_cnn_gru,
        'Transformer': train_transformer_model,
        'TimeDistributed': train_time_distributed_model
    }
    if ml_model in model_functions:
        preds = model_functions[ml_model](price_series, lookback, forecast_days)
        if preds is None or len(preds) != forecast_days:
            logger.error(
                f"Model {ml_model} returned {len(preds) if preds is not None else 'None'} predictions, expected {forecast_days}")
            return np.zeros(forecast_days)
        logger.info(f"Model {ml_model} output length: {len(preds)} (Expected: {forecast_days})")
        return preds
    else:
        logger.warning(f"Model {ml_model} not implemented; falling back to LSTM")
        preds = train_lstm_model(price_series, lookback, forecast_days)
        if preds is None or len(preds) != forecast_days:
            logger.error(
                f"Fallback LSTM returned {len(preds) if preds is not None else 'None'} predictions, expected {forecast_days}")
            return np.zeros(forecast_days)
        logger.info(f"Fallback LSTM output length: {len(preds)} (Expected: {forecast_days})")
        return preds

def monte_carlo_simulation(returns, initial_value, lstm_predictions=None, days=252, simulations=10000):
    try:
        if lstm_predictions is not None:
            lstm_predictions = np.clip(lstm_predictions, a_min=1e-6, a_max=None)
            log_returns = np.diff(np.log(lstm_predictions))
            if np.any(np.isnan(log_returns)) or np.any(np.isinf(log_returns)):
                logger.warning("Invalid log returns from LSTM predictions; falling back to historical returns")
                mu = returns.mean()
                sigma = returns.std()
            else:
                mu = np.mean(log_returns)
                sigma = np.std(log_returns)
        else:
            mu = returns.mean()
            sigma = returns.std()
        paths = np.zeros((simulations, days + 1))
        paths[:, 0] = initial_value
        dt = 1
        for t in range(1, days + 1):
            random_shocks = np.random.normal(0, 1, simulations)
            paths[:, t] = paths[:, t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * random_shocks)
        percentiles = {
            'median': np.percentile(paths, 50, axis=0),
            'lower_band': np.percentile(paths, 5, axis=0),
            'upper_band': np.percentile(paths, 95, axis=0)
        }
        return paths, percentiles
    except Exception as e:
        logger.error(f"Error in Monte Carlo simulation: {str(e)}")
        return None, None


def create_monte_carlo_chart(total_holdings, monte_carlo_paths, percentiles, end_date, portfolio_index):
    try:
        last_date = pd.to_datetime(list(total_holdings.index)[-1])
        future_dates = [last_date + timedelta(days=i) for i in range(len(percentiles['median']))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=total_holdings.index,
            y=total_holdings.values,
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))
        sample_size = min(100, monte_carlo_paths.shape[0])
        sample_paths = monte_carlo_paths[np.random.choice(monte_carlo_paths.shape[0], sample_size, replace=False)]
        for path in sample_paths:
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=path,
                mode='lines',
                line=dict(color='rgba(200, 200, 200, 0.2)'),
                showlegend=False,
                hoverinfo='skip'
            ))
        fig.add_trace(go.Scatter(x=future_dates, y=percentiles['upper_band'], mode='lines', name='95th Percentile',
                                 line=dict(color='red', dash='dash')))
        fig.add_trace(
            go.Scatter(x=future_dates, y=percentiles['median'], mode='lines', name='Median', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=future_dates, y=percentiles['lower_band'], mode='lines', name='5th Percentile',
                                 line=dict(color='red', dash='dash')))
        fig.update_layout(
            title=f'Portfolio {portfolio_index + 1} Monte Carlo Simulation (252-day Projection)',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            height=600,
            width=1200,
            showlegend=True,
            template='plotly_white'
        )
        chart_path = os.path.join(STATIC_DIR, f'monte_carlo_{portfolio_index}.html')
        fig.write_html(chart_path)
        return f'monte_carlo_{portfolio_index}.html'
    except Exception as e:
        logger.error(f"Error creating Monte Carlo chart: {str(e)}")
        return None


def create_lstm_prediction_chart(crypto_prices, predictions_dict, end_date, portfolio_index):
    try:
        last_date = pd.to_datetime(list(crypto_prices.index)[-1])
        future_dates = [last_date + timedelta(days=i) for i in range(252 + 1)][1:]
        fig = go.Figure()
        for crypto in crypto_prices.columns:
            fig.add_trace(go.Scatter(
                x=crypto_prices.index,
                y=crypto_prices[crypto],
                mode='lines',
                name=f'{crypto} Historical',
                line=dict(width=2)
            ))
        for crypto, preds in predictions_dict.items():
            if preds is not None:
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=preds,
                    mode='lines',
                    name=f'{crypto} Predicted',
                    line=dict(dash='dash', width=2)
                ))
        fig.update_layout(
            title=f'Portfolio {portfolio_index + 1} LSTM Price Predictions (252-day Projection)',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            height=600,
            width=1200,
            showlegend=True,
            template='plotly_white'
        )
        chart_path = os.path.join(STATIC_DIR, f'lstm_prediction_{portfolio_index}.html')
        fig.write_html(chart_path)
        return f'lstm_prediction_{portfolio_index}.html'
    except Exception as e:
        logger.error(f"Error creating LSTM chart: {str(e)}")
        return None


def save_portfolio_visualizations(index, total_holdings, detailed_portfolio, correlation_matrix, cryptos,
                                  interactive_path, correlation_path, interactive_dist):
    try:
        # Interactive Portfolio Performance (HTML)
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Portfolio Performance", "Individual Holdings",
                                                            "Returns Distribution", "Q-Q Plot"))
        fig.add_trace(go.Scatter(x=total_holdings.index, y=total_holdings.values, mode='lines', name='Portfolio'),
                      row=1, col=1)
        for crypto in cryptos:
            crypto_data = detailed_portfolio[detailed_portfolio['Crypto'] == crypto]
            fig.add_trace(go.Scatter(x=crypto_data['Date'], y=crypto_data['Holdings'], mode='lines', name=crypto),
                          row=1, col=2)
        returns = pd.Series(total_holdings).pct_change().dropna()
        fig.add_trace(go.Histogram(x=returns, nbinsx=30, name='Returns', histnorm='probability density'), row=2, col=1)
        kde_x = np.linspace(min(returns), max(returns), 100)
        kde = stats.gaussian_kde(returns)
        fig.add_trace(go.Scatter(x=kde_x, y=kde(kde_x), name='KDE', line=dict(color='red')), row=2, col=1)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(returns)))
        observed_quantiles = np.sort(returns)
        fig.add_trace(go.Scatter(x=theoretical_quantiles, y=observed_quantiles, mode='markers', name='Q-Q Plot'),
                      row=2, col=2)
        min_val, max_val = min(theoretical_quantiles), max(theoretical_quantiles)
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Reference Line',
                                 line=dict(color='red', dash='dash')), row=2, col=2)
        fig.update_layout(height=1200, width=1200, title_text="Portfolio Analysis Dashboard", showlegend=True)
        fig.write_html(interactive_path)

        # Individual Holdings (PNG using Matplotlib)
        plt.figure(figsize=(8, 4))
        for crypto in cryptos:
            crypto_data = detailed_portfolio[detailed_portfolio['Crypto'] == crypto]
            plt.plot(crypto_data['Date'], crypto_data['Holdings'], label=crypto)
        plt.title(f'Portfolio {index + 1} Individual Holdings')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.tight_layout()
        individual_image_path = os.path.join(STATIC_DIR, f'portfolio_{index}_holdings.png')
        plt.savefig(individual_image_path)
        plt.close()

        # Correlation Matrix (PNG using Matplotlib)
        plt.figure(figsize=(8, 6))
        plt.imshow(correlation_matrix.values, cmap='RdBu', vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
        plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
        for i in range(len(correlation_matrix.index)):
            for j in range(len(correlation_matrix.columns)):
                plt.text(j, i, f"{correlation_matrix.values[i, j]:.2f}", ha='center', va='center', color='black')
        plt.title(f'Portfolio {index + 1} Cryptocurrency Price Correlation')
        correlation_image_path = os.path.join(STATIC_DIR, f'portfolio_{index}_correlation.png')
        plt.savefig(correlation_image_path)
        plt.close()

        # Interactive Distribution (HTML)
        fig_dist = make_subplots(rows=1, cols=2, subplot_titles=(f'Portfolio {index + 1} Returns Distribution',
                                                                f'Portfolio {index + 1} Q-Q Plot'))
        fig_dist.add_trace(go.Histogram(x=returns, nbinsx=30, name='Returns', histnorm='probability density'),
                          row=1, col=1)
        fig_dist.add_trace(go.Scatter(x=kde_x, y=kde(kde_x), name='KDE', line=dict(color='red')), row=1, col=1)
        fig_dist.add_trace(go.Scatter(x=theoretical_quantiles, y=observed_quantiles, mode='markers', name='Q-Q Plot'),
                          row=1, col=2)
        fig_dist.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Reference Line',
                                     line=dict(color='red', dash='dash')), row=1, col=2)
        fig_dist.update_layout(height=500, width=1200, showlegend=True,
                              title_text=f"Portfolio {index + 1} Return Distribution Analysis")
        fig_dist.write_html(interactive_dist)

        # Individual Holdings (Interactive HTML)
        fig_individual = go.Figure()
        for crypto in cryptos:
            crypto_data = detailed_portfolio[detailed_portfolio['Crypto'] == crypto]
            fig_individual.add_trace(go.Scatter(x=crypto_data['Date'], y=crypto_data['Holdings'], mode='lines',
                                               name=crypto))
        fig_individual.update_layout(title=f'Portfolio {index + 1} Individual Holdings', xaxis_title='Date',
                                     yaxis_title='Value ($)', height=400, width=800)
        fig_individual.write_html(os.path.join(STATIC_DIR, f'individual_holdings_{index}.html'))

        # Correlation Matrix (Interactive HTML)
        fig_corr = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 12}
        ))
        fig_corr.update_layout(title=f'Portfolio {index + 1} Cryptocurrency Price Correlation', height=600, width=800)
        fig_corr.write_html(os.path.join(STATIC_DIR, f'correlation_matrix_{index}.html'))

        logger.info(f"Visualizations saved for portfolio {index}: {individual_image_path}, {correlation_image_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving visualizations for portfolio {index}: {str(e)}")
        return False


def create_distribution_plots(returns, portfolio_index):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
    f'Portfolio {portfolio_index + 1} Returns Distribution', f'Portfolio {portfolio_index + 1} Q-Q Plot'))
    fig.add_trace(go.Histogram(x=returns, nbinsx=30, name='Returns', histnorm='probability density'), row=1, col=1)
    kde_x = np.linspace(min(returns), max(returns), 100)
    kde = stats.gaussian_kde(returns)
    fig.add_trace(go.Scatter(x=kde_x, y=kde(kde_x), name='KDE', line=dict(color='red')), row=1, col=1)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(returns)))
    observed_quantiles = np.sort(returns)
    fig.add_trace(go.Scatter(x=theoretical_quantiles, y=observed_quantiles, mode='markers', name='Q-Q Plot'), row=1,
                  col=2)
    min_val, max_val = min(theoretical_quantiles), max(theoretical_quantiles)
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Reference Line',
                             line=dict(color='red', dash='dash')), row=1, col=2)
    fig.update_layout(height=500, width=1200, showlegend=True,
                      title_text=f"Portfolio {portfolio_index + 1} Return Distribution Analysis")
    interactive_dist_path = os.path.join(STATIC_DIR, f'portfolio_{portfolio_index}_distribution_interactive.html')
    fig.write_html(interactive_dist_path)
    return f'portfolio_{portfolio_index}_distribution_interactive.html'

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)

