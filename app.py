from flask import Flask, render_template, request, jsonify, url_for, redirect
import requests
from binance.client import Client as BinanceClient
from kucoin.client import Market as KucoinMarket
import yfinance as yf
import uuid
import redis
from datetime import datetime
import json
from celery import Celery
import os
import logging
import pandas as pd
import numpy as np
from scipy import stats
import traceback
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras import Model, Input
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Bidirectional, Dropout, Dense, Conv1D, MaxPooling1D, LayerNormalization, \
    MultiHeadAttention, Add, GlobalAveragePooling1D, TimeDistributed
from keras.src.optimizers import Adam


# Initialize Flask app
app = Flask(__name__)

# Disable GPU for TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Use Render's PORT environment variable
port = int(os.environ.get('PORT', 5000))  # Changed default to 5000

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
    # Continue without failing since the app might work without Redis

# Celery configuration without SSL
app.config.update(
    CELERY_BROKER_URL=redis_url,
    CELERY_RESULT_BACKEND=redis_url,
    BROKER_CONNECTION_RETRY=True,
    BROKER_CONNECTION_RETRY_ON_STARTUP=True
)

# Initialize Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'], backend=app.config['CELERY_RESULT_BACKEND'])
celery.conf.update(app.config)

# Binance API client - Use environment variables for credentials
binance_api_key = os.getenv('BINANCE_API_KEY', 'enter_your_key')
binance_api_secret = os.getenv('BINANCE_API_SECRET', 'enter_your_secret_password')
binance_client = BinanceClient(binance_api_key, binance_api_secret)

# KuCoin API client
kucoin_api_key = os.getenv('KUCOIN_API_KEY', 'enter_your_key')
kucoin_api_secret = os.getenv('KUCOIN_API_SECRET', 'enter_your_secret_password')
kucoin_api_passphrase = os.getenv('KUCOIN_API_PASSPHRASE', 'enter_your_passphrase')
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
def process_portfolio_task(self, cryptos, allocations, start_date, end_date, initial_capital, portfolio_index, task_id, ml_model='LSTM'):
    import tensorflow as tf
    logger.info(f"Starting background processing for portfolio {portfolio_index} with model {ml_model}")
    task_status_path = os.path.join(TASK_DIR, f"{task_id}.json")
    filename = f"{ml_model.lower()}_prediction_{portfolio_index}.html"

    update_task_status(task_id, {
        'portfolio_index': portfolio_index,
        'status': 'PROCESSING',
        'progress': 0,
        'message': f'Starting analysis for portfolio {portfolio_index + 1}',
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

            # Explicitly set forecast_days to 252 to match simulation length
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

        # Create visualizations
        portfolio_image = f'portfolio_{portfolio_index}_performance.png'
        individual_image = f'portfolio_{portfolio_index}_holdings.png'
        correlation_matrix_image = f'portfolio_{portfolio_index}_correlation.png'
        interactive_portfolio = f'interactive_portfolio_{portfolio_index}.html'
        distribution_image = f'portfolio_{portfolio_index}_distribution.png'
        interactive_dist = f'Portfolio_{portfolio_index}_distribution_interactive.html'

        save_portfolio_visualizations(
            portfolio_index, total_holdings, detailed_portfolio, correlation_matrix, cryptos,
            os.path.join(STATIC_DIR, portfolio_image),
            os.path.join(STATIC_DIR, individual_image),
            os.path.join(STATIC_DIR, correlation_matrix_image),
            os.path.join(STATIC_DIR, interactive_portfolio),
            os.path.join(STATIC_DIR, distribution_image),
            os.path.join(STATIC_DIR, interactive_dist)
        )

        # Update progress
        update_task_status(task_id, {
            'progress': 80,
            'message': 'Running Monte Carlo simulations...'
        })

        # Monte Carlo Simulation with model predictions
        returns = total_holdings.pct_change().dropna()
        mc_paths, mc_percentiles = monte_carlo_simulation(
            returns,
            final_value,
            lstm_predictions=predicted_total_holdings.values if not predicted_total_holdings.empty else None
        )
        mc_chart = None
        if mc_paths is not None and mc_percentiles is not None:
            mc_chart = create_monte_carlo_chart(total_holdings, mc_paths, mc_percentiles, end_date, portfolio_index)

        # Prediction Chart (renamed from lstm_chart to model_prediction_chart for flexibility)
        model_prediction_chart = create_model_prediction_chart(crypto_prices, predictions_dict, end_date,
                                                               portfolio_index, ml_model)

        portfolio_data = {
            'total_holdings': {str(k): float(v) for k, v in total_holdings.to_dict().items()},
            'initial_capital': float(initial_capital),
            'final_value': float(final_value),
            'performance': float(performance),
            'financial_metrics': {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()},
            'portfolio_image': portfolio_image,
            'individual_image': individual_image,
            'correlation_matrix_image': correlation_matrix_image,
            'interactive_portfolio': interactive_portfolio,
            'distribution_image': distribution_image,
            'interactive_distribution': interactive_dist,
            'monte_carlo_chart': mc_chart,
            'prediction_chart': model_prediction_chart,
            'ml_model': ml_model  # Store the model type used
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

def create_model_prediction_chart(crypto_prices, predictions_dict, end_date, portfolio_index, model_name='LSTM'):
    """Create an interactive chart showing historical prices and model predictions"""
    try:
        last_date = pd.to_datetime(list(crypto_prices.index)[-1])
        future_dates = [last_date + timedelta(days=i) for i in range(252 + 1)][1:]
        fig = go.Figure()

        # Historical data for each crypto
        for crypto in crypto_prices.columns:
            fig.add_trace(go.Scatter(
                x=crypto_prices.index,
                y=crypto_prices[crypto],
                mode='lines',
                name=f'{crypto} Historical',
                line=dict(width=2)
            ))

        # Predicted data for each crypto
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
    """Update the task status in Redis"""
    try:
        # Fetch existing data to preserve fields like initial_capital
        current_data = get_task_status(task_id) or {}
        # Merge current data with updates, ensuring initial_capital persists
        updated_data = {**current_data, **update_data}
        r.set(f"task:{task_id}", json.dumps(updated_data))
        logger.debug(f"Updated task {task_id} status: {updated_data.get('status', '')}, progress: {updated_data.get('progress', '')}")
    except Exception as e:
        logger.error(f"Error updating task status in Redis: {str(e)}")

@app.route('/task_status/<task_id>')
def task_status(task_id):
    """API endpoint to get the status of a specific task"""
    status = get_task_status(task_id)
    if status:
        # Ensure initial_capital is always returned, fallback to current value if missing
        if 'initial_capital' not in status:
            status['initial_capital'] = "0.00"  # Fallback if not set
        return jsonify(status)
    return jsonify({'error': 'Task not found'}), 404


def get_task_status(task_id):
    """Get the current status of a task from Redis"""
    try:
        data = r.get(f"task:{task_id}")
        return json.loads(data) if data else None
    except Exception as e:
        logger.error(f"Error reading task status from Redis: {str(e)}")
        return None


def save_simulation(portfolio_id, data):
    try:
        # Convert total_holdings keys (Timestamps) to strings
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
        for key in r.scan_iter("portfolio:*"):  # Replace r.keys()
            data = r.get(key)
            if data:
                sim_data = json.loads(data)
                if 'total_holdings' in sim_data:
                    sim_data['total_holdings'] = pd.Series(sim_data['total_holdings'])
                simulations.append(sim_data)
    except Exception as e:
        logger.error(f"Error loading simulations from Redis: {str(e)}")
    return simulations


# Keep all your existing functions here, unchanged
# calculate_financial_metrics, fetch_market_cap, format_portfolio_for_template,
# get_historical_data, train_lstm_model, create_lstm_prediction_chart,
# monte_carlo_simulation, create_monte_carlo_chart, etc.

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
            'portfolio_image': portfolio_data.get('portfolio_image', ''),
            'individual_image': portfolio_data.get('individual_image', ''),
            'correlation_matrix_image': portfolio_data.get('correlation_matrix_image', ''),
            'interactive_portfolio': portfolio_data.get('interactive_portfolio', ''),
            'distribution_image': portfolio_data.get('distribution_image', ''),
            'interactive_distribution': portfolio_data.get('interactive_distribution', ''),
            'monte_carlo_chart': portfolio_data.get('monte_carlo_chart', ''),
            'prediction_chart': portfolio_data.get('prediction_chart', '') # Added LSTM field
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

            # Fetch from KuCoin
            try:
                klines = kucoin_client.get_kline(symbol=kucoin_symbol, kline_type='1day', startAt=start_timestamp, endAt=end_timestamp)
                if klines and len(klines) > 0:
                    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume', 'amount'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
                    df.set_index('timestamp', inplace=True)
                    df = df.sort_index()
                    price_data_kucoin = df['close'].astype(float)
                    logger.info(f"KuCoin data for {crypto} starts at {price_data_kucoin.index[0]} and ends at {price_data_kucoin.index[-1]}")
                    if start_cut_kucoin <= price_data_kucoin.index[0]:
                        start_cut_kucoin = price_data_kucoin.index[0]
                    logger.info(f"start_cut_kucoin: {start_cut_kucoin}")
                else:
                    logger.warning(f"No KuCoin data for {kucoin_symbol}")
            except Exception as e:
                logger.error(f"Error fetching KuCoin data for {kucoin_symbol}: {str(e)}")

            # Fetch from yFinance
            try:
                ticker = yf.Ticker(f"{crypto}-USD")
                yf_data = ticker.history(start=start_date, end=end_date, interval='1d')
                if not yf_data.empty:
                    price_data_yfinance = yf_data['Close']
                    price_data_yfinance.index = pd.to_datetime(price_data_yfinance.index).tz_localize(None)
                    logger.info(f"yFinance data for {crypto} starts at {price_data_yfinance.index[0]} and ends at {price_data_yfinance.index[-1]}")
                    if start_cut_yf <= price_data_yfinance.index[0]:
                        start_cut_yf = price_data_yfinance.index[0]
                    logger.info(f"start_cut_yfi: {start_cut_yf}")
                else:
                    ticker = yf.Ticker(crypto)
                    yf_data = ticker.history(start=start_date, end=end_date, interval='1d')
                    if not yf_data.empty:
                        price_data_yfinance = yf_data['Close']
                        price_data_yfinance.index = pd.to_datetime(price_data_yfinance.index).tz_localize(None)
                        logger.info(f"yFinance data for {crypto} (alt ticker) starts at {price_data_yfinance.index[0]} and ends at {price_data_yfinance.index[-1]}")
                    else:
                        logger.warning(f"No yFinance data for {crypto}")
            except Exception as e:
                logger.error(f"Error fetching yFinance data for {crypto}: {str(e)}")

            start_cut = min(start_cut_kucoin, start_cut_yf)  # Your original logic
            price_data_list = [data for data in [price_data_kucoin, price_data_yfinance] if data is not None]
            if not price_data_list:
                logger.error(f"No data available for {crypto} from any source")
                continue

            # Combine and reindex price_data
            combined_data = pd.concat(price_data_list).sort_index()
            price_data = combined_data.groupby(combined_data.index).mean()
            full_index = pd.date_range(start=start_date, end=end_date, freq='1D')
            price_data = price_data.reindex(full_index, method=None)

            # Determine earliest date with non-NaN price data
            price_data_non_na = price_data.dropna()
            if price_data_non_na.empty:
                logger.error(f"No valid price data for {crypto} after reindexing")
                continue
            earliest_price_date = price_data_non_na.index[0]
            logger.info(f"{crypto} earliest price date: {earliest_price_date}")

            # Filter price_data to start at the later of start_cut and earliest_price_date
            price_data = price_data[price_data.index >= max(start_cut, earliest_price_date)]
            price_data = price_data.ffill()  # Fill forward any remaining gaps
            logger.info(f"{crypto} price_data after trimming: shape={price_data.shape}, first={price_data.index[0]}, last={price_data.index[-1]}")

            # Calculate holdings
            initial_investment = initial_capital * allocations[i]
            first_valid_price = price_data.iloc[0]  # Now guaranteed to be non-NaN
            num_coins = initial_investment / float(first_valid_price)
            holdings = pd.DataFrame({
                'Date': price_data.index,
                'Crypto': crypto,
                'Holdings': (price_data * num_coins),
                'Initial_Investment': initial_investment
            })
            logger.info(f"{crypto} Holdings: first={holdings['Holdings'].iloc[0]}, last={holdings['Holdings'].iloc[-1]}, NaN count={holdings['Holdings'].isna().sum()}")
            portfolio_data.append(holdings)
            crypto_prices[crypto] = price_data

        if not portfolio_data:
            raise ValueError("No data could be retrieved for any cryptocurrency")

        filtered_data = []
        for i, df in enumerate(portfolio_data):
            df = df[df['Date'] >= start_cut]  # Your original filtering
            logger.info(f"Portfolio {i} shape: {df.shape}, columns: {list(df.columns)}, first date: {df['Date'].min()}, last date: {df['Date'].max()}")
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


# Only modifying the routes below:

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
        allocations = request.form  # e.g., {'BTC': '50', 'ETH': '30', 'ADA': '20'}
        alloc_str = ", ".join(f"{coin}={pct}%" for coin, pct in allocations.items())
        logger.info(f"User submitted crypto allocations: {alloc_str}")
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        portfolio_count = int(request.form.get('portfolio_count', 0))
        ml_model = request.form.get('ml_model', 'LSTM')  # Default to LSTM if not provided
        logger.info(f"Processing {portfolio_count} portfolios from {start_date} to {end_date} with model {ml_model}")

        batch_id = str(uuid.uuid4())
        tasks = []

        for i in range(portfolio_count):
            try:
                cryptos = request.form.getlist(f'crypto_{i}[]')
                allocations = [float(request.form.get(f'portfolio_{i}_allocation_{crypto}', 0)) / 100 for crypto in
                               cryptos]
                initial_capital = float(
                    request.form.get(f'initial_capital_{i}', request.form.get('initial_capital')))

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

                # Pass ml_model to the Celery task
                process_portfolio_task.delay(
                    cryptos,
                    allocations,
                    start_date,
                    end_date,
                    initial_capital,
                    i,
                    task_id,
                    ml_model  # Add ml_model here
                )

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
            'ml_model': ml_model,  # Store ml_model in batch data
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
    """Display the status of a batch of portfolio simulations"""
    try:
        # Load batch information
        batch_file = os.path.join(TASK_DIR, f"{batch_id}_batch.json")
        if not os.path.exists(batch_file):
            return render_template('error.html', error="Batch not found")

        with open(batch_file, 'r') as f:
            batch_data = json.load(f)

        # Get status of all tasks
        task_statuses = []
        for task_id in batch_data['tasks']:
            status = get_task_status(task_id)
            if status:
                # Ensure portfolio_index exists in the status
                if 'portfolio_index' not in status:
                    # Extract portfolio_index from task_id if possible, or use a default
                    parts = task_id.split('_')
                    if len(parts) > 1:
                        try:
                            status['portfolio_index'] = int(parts[-1])
                        except ValueError:
                            status['portfolio_index'] = -1
                    else:
                        status['portfolio_index'] = -1
                task_statuses.append(status)

        # Calculate overall progress
        completed_tasks = sum(1 for task in task_statuses if task.get('status') == 'COMPLETED')
        failed_tasks = sum(1 for task in task_statuses if task.get('status') == 'FAILED')
        total_tasks = len(batch_data['tasks'])
        overall_progress = int(
            (sum(task.get('progress', 0) for task in task_statuses) / total_tasks) if total_tasks > 0 else 0)

        # Check if all tasks are completed
        all_completed = (completed_tasks + failed_tasks) == total_tasks

        return render_template(
            'status.html',
            batch_id=batch_id,
            batch_data=batch_data,
            task_statuses=task_statuses,
            overall_progress=overall_progress,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            total_tasks=total_tasks,
            all_completed=all_completed
        )

    except Exception as e:
        error_msg = f"Error checking status: {str(e)}"
        logger.error(error_msg)
        return render_template('error.html', error=error_msg)

@app.route('/results/<batch_id>')
def results(batch_id):
    try:
        batch_file = os.path.join(TASK_DIR, f"{batch_id}_batch.json")
        if not os.path.exists(batch_file):
            return render_template('error.html', error="Batch not found")

        with open(batch_file, 'r') as f:
            batch_data = json.load(f)

        task_statuses = []
        for task_id in batch_data['tasks']:
            status = get_task_status(task_id)
            if status:
                task_statuses.append(status)

        completed_tasks = sum(1 for task in task_statuses if task.get('status') == 'COMPLETED')
        failed_tasks = sum(1 for task in task_statuses if task.get('status') == 'FAILED')
        total_tasks = len(batch_data['tasks'])

        if completed_tasks + failed_tasks < total_tasks:
            return redirect(url_for('status', batch_id=batch_id))

        portfolios = []
        errors = []

        for task in task_statuses:
            if task.get('status') == 'COMPLETED' and task.get('result'):
                portfolio_data = task.get('result')
                formatted_portfolio = format_portfolio_for_template(portfolio_data)
                if formatted_portfolio:
                    portfolios.append(portfolio_data)
                else:
                    logger.warning(f"Portfolio {task.get('portfolio_index', '?')} formatted as None")
                    errors.append(f"Formatting failed for portfolio {task.get('portfolio_index', '?')}")
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
            start_date=batch_data['start_date'],
            end_date=batch_data['end_date'],
            portfolios=formatted_portfolios,
            portfolio_correlation=portfolio_correlation,
            combined_chart=combined_chart,
            combined_interactive=combined_interactive,
            heatmap_filename=heatmap_filename,
            error="\n".join(errors) if errors else None,
            ml_model=batch_data.get('ml_model', 'LSTM')
        )

    except Exception as e:
        error_msg = f"Error displaying results: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return render_template('error.html', error=error_msg)

@app.route('/donate')
def donate():
    return render_template('donate.html')


def save_correlation_heatmap(portfolio_correlation, filename='correlation_heatmap.png'):
    if not isinstance(portfolio_correlation, pd.DataFrame):
        portfolio_correlation = pd.DataFrame(portfolio_correlation)
    plt.figure(figsize=(10, 8))
    sns.heatmap(portfolio_correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Portfolio Correlation Heatmap')
    static_path = os.path.join(STATIC_DIR, filename)
    plt.savefig(static_path, bbox_inches='tight')
    plt.close()
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
    """Preprocess time series data for model training"""
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(price_series.values.reshape(-1, 1))

    # Create sequences
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Train-test split
    train_size = int(len(X) * (1 - test_split))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test, scaler, scaled_data


def train_gru_model(price_series, lookback=60, forecast_days=252):
    from tensorflow.keras import Sequential, Input
    from tensorflow.keras.layers import Dropout, Dense, GRU
    from keras.src.optimizers import Adam
    """Train and generate predictions using a GRU (Gated Recurrent Unit) model"""
    try:
        X_train, X_test, y_train, y_test, scaler, scaled_data = preprocess_data(price_series, lookback, forecast_days)

        # Build GRU model
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

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train the model
        model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )

        # Generate future predictions
        future_predictions = generate_predictions(model, scaled_data, lookback, forecast_days, scaler)
        return future_predictions

    except Exception as e:
        logger.error(f"Error training GRU model: {str(e)}")
        return None


def train_bidirectional_rnn(price_series, lookback=60, forecast_days=252, rnn_type='LSTM'):
    from tensorflow.keras import Sequential, Input
    from tensorflow.keras.layers import LSTM, Dropout, Dense, GRU
    from keras.src.optimizers import Adam
    """Train and generate predictions using a Bidirectional RNN (LSTM or GRU)"""
    try:
        X_train, X_test, y_train, y_test, scaler, scaled_data = preprocess_data(price_series, lookback, forecast_days)

        # Choose RNN cell type
        rnn_layer = LSTM if rnn_type.upper() == 'LSTM' else GRU

        # Build Bidirectional RNN model
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

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train the model
        model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )

        # Generate future predictions
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
    """Train and generate predictions using a 1D Convolutional Neural Network"""
    try:
        X_train, X_test, y_train, y_test, scaler, scaled_data = preprocess_data(price_series, lookback, forecast_days)

        # Build 1D CNN model

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

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train the model
        model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )

        # Generate future predictions
        future_predictions = generate_predictions(model, scaled_data, lookback, forecast_days, scaler)
        return future_predictions

    except Exception as e:
        logger.error(f"Error training 1D CNN model: {str(e)}")
        return None


def train_cnn_gru(price_series, lookback=60, forecast_days=252):
    from tensorflow.keras import Input
    from tensorflow.keras.layers import Dropout, Dense, GRU
    from keras.src.optimizers import Adam
    """Train and generate predictions using a CNN-GRU hybrid model"""
    try:
        X_train, X_test, y_train, y_test, scaler, scaled_data = preprocess_data(price_series, lookback, forecast_days)

        # Build CNN-GRU hybrid model
        input_layer = Input(shape=(lookback, 1))

        # CNN block for feature extraction
        cnn = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        cnn = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(cnn)

        # GRU block for temporal dependencies
        gru = GRU(units=64, return_sequences=True)(cnn)
        gru = Dropout(0.2)(gru)
        gru = GRU(units=32, return_sequences=False)(gru)
        gru = Dropout(0.2)(gru)

        # Output block
        dense = Dense(units=16)(gru)
        output = Dense(units=1)(dense)

        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train the model
        model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )

        # Generate future predictions
        future_predictions = generate_predictions(model, scaled_data, lookback, forecast_days, scaler)
        return future_predictions

    except Exception as e:
        logger.error(f"Error training CNN-GRU model: {str(e)}")
        return None


def train_transformer_model(price_series, lookback=60, forecast_days=252):

    from keras.src.optimizers import Adam
    """Train and generate predictions using a Transformer model"""
    try:
        X_train, X_test, y_train, y_test, scaler, scaled_data = preprocess_data(price_series, lookback, forecast_days)

        # Build Transformer model
        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
            # Multi-head attention
            x = LayerNormalization(epsilon=1e-6)(inputs)
            x = MultiHeadAttention(
                key_dim=head_size, num_heads=num_heads, dropout=dropout
            )(x, x)
            x = Dropout(dropout)(x)
            res = Add()([x, inputs])

            # Feed forward
            x = LayerNormalization(epsilon=1e-6)(res)
            x = Dense(ff_dim, activation="relu")(x)
            x = Dropout(dropout)(x)
            x = Dense(inputs.shape[-1])(x)
            return Add()([x, res])

        from keras import Input
        input_layer = Input(shape=(lookback, 1))

        # Transformer encoder blocks
        x = input_layer
        for _ in range(2):  # Number of transformer blocks
            x = transformer_encoder(x, head_size=32, num_heads=2, ff_dim=64, dropout=0.2)

        # Global pooling
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation="relu")(x)
        x = Dropout(0.2)(x)
        output = Dense(1)(x)

        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train the model
        model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )

        # Generate future predictions
        future_predictions = generate_predictions(model, scaled_data, lookback, forecast_days, scaler)
        return future_predictions

    except Exception as e:
        logger.error(f"Error training Transformer model: {str(e)}")
        return None


def train_time_distributed_model(price_series, lookback=60, forecast_days=252):


    """Train and generate predictions using a TimeDistributed model"""
    try:
        X_train, X_test, y_train, y_test, scaler, scaled_data = preprocess_data(price_series, lookback, forecast_days)

        # Reshape input for TimeDistributed (splitting the time steps into sub-sequences)
        sub_seq_len = 5  # Length of sub-sequences
        if lookback % sub_seq_len != 0:
            # Adjust lookback to be divisible by sub_seq_len
            new_lookback = (lookback // sub_seq_len) * sub_seq_len
            X_train = X_train[:, -new_lookback:, :]
            X_test = X_test[:, -new_lookback:, :]
            lookback = new_lookback

        X_train = X_train.reshape(X_train.shape[0], lookback // sub_seq_len, sub_seq_len, 1)
        X_test = X_test.reshape(X_test.shape[0], lookback // sub_seq_len, sub_seq_len, 1)

        # Build TimeDistributed model
        input_layer = Input(shape=(lookback // sub_seq_len, sub_seq_len, 1))

        # Apply the same Conv1D to each sub-sequence
        x = TimeDistributed(Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'))(input_layer)
        x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
        from keras.src.layers import Flatten
        x = TimeDistributed(Flatten())(x)

        # Process the sequence of outputs
        from keras.src.layers import LSTM
        x = LSTM(64, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        output = Dense(1)(x)

        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train the model
        model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )

        # For prediction, we need to adapt our generate_predictions function to handle the TimeDistributed structure
        # We'll create a custom prediction function for this specific model

        last_sequence = scaled_data[-lookback:]
        future_predictions = []

        for _ in range(forecast_days):
            # Reshape the sequence for TimeDistributed
            sequence_reshaped = last_sequence.reshape(1, lookback // sub_seq_len, sub_seq_len, 1)

            # Predict next value
            next_pred = model.predict(sequence_reshaped, verbose=0)

            # Append to predictions
            future_predictions.append(next_pred[0, 0])

            # Update last_sequence (remove oldest, add newest)
            last_sequence = np.append(last_sequence[1:], next_pred)

        # Inverse transform the predictions
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # Avoid zero/negative values
        future_predictions = np.clip(future_predictions, a_min=1e-6, a_max=None)

        return future_predictions.flatten()

    except Exception as e:
        logger.error(f"Error training TimeDistributed model: {str(e)}")
        return None


def generate_predictions(model, scaled_data, lookback, forecast_days, scaler):
    """Generate future predictions using the trained model"""
    last_sequence = scaled_data[-lookback:]
    future_predictions = []

    for _ in range(forecast_days):
        last_sequence_reshaped = last_sequence.reshape((1, lookback, 1))
        next_pred = model.predict(last_sequence_reshaped, verbose=0)
        future_predictions.append(next_pred[0, 0])
        last_sequence = np.append(last_sequence[1:], next_pred)

    # Inverse transform
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Avoid zero/negative values
    future_predictions = np.clip(future_predictions, a_min=1e-6, a_max=None)

    return future_predictions.flatten()


def select_model_for_prediction(price_series, ml_model, lookback=60, forecast_days=252):
    logger.info(f"Training {ml_model} model for price prediction")

    model_functions = {
        'LSTM': train_lstm_model,
        'GRU': train_gru_model,
        'BiLSTM': lambda price_series, lookback, forecast_days: train_bidirectional_rnn(price_series, lookback, forecast_days, rnn_type='LSTM'),
        'BiGRU': lambda price_series, lookback, forecast_days: train_bidirectional_rnn(price_series, lookback, forecast_days, rnn_type='GRU'),
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
            # Fallback to zeros or raise an exception
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
            # Filter out invalid values and ensure positivity
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

        # Rest of the Monte Carlo logic
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
    """Create an interactive chart showing historical prices and LSTM predictions"""
    try:
        last_date = pd.to_datetime(list(crypto_prices.index)[-1])
        future_dates = [last_date + timedelta(days=i) for i in range(252 + 1)][1:]
        fig = go.Figure()

        # Historical data for each crypto
        for crypto in crypto_prices.columns:
            fig.add_trace(go.Scatter(
                x=crypto_prices.index,
                y=crypto_prices[crypto],
                mode='lines',
                name=f'{crypto} Historical',
                line=dict(width=2)
            ))

        # Predicted data for each crypto
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
                                  portfolio_path, individual_path, correlation_path, interactive_path,
                                  distribution_image, interactive_dist):
    try:
        logger.info(
            f"Plotting total_holdings: first non-NaN date {total_holdings.dropna().index[0]}, last date {total_holdings.index[-1]}")
        plt.figure(figsize=(12, 6))
        plt.plot(total_holdings.index, total_holdings.values)
        plt.title('Portfolio Investment Performance')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.savefig(portfolio_path)
        plt.close()

        plt.figure(figsize=(12, 6))
        for crypto in cryptos:
            crypto_data = detailed_portfolio[detailed_portfolio['Crypto'] == crypto]
            plt.plot(crypto_data['Date'], crypto_data['Holdings'], label=crypto)
        plt.title('Individual Cryptocurrency Holdings')
        plt.xlabel('Date')
        plt.ylabel('Holdings Value ($)')
        plt.legend()
        plt.grid(True)
        plt.savefig(individual_path)
        plt.close()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True, linewidths=0.5)
        plt.title('Cryptocurrency Price Correlation Heatmap')
        plt.savefig(correlation_path)
        plt.close()

        returns = pd.Series(total_holdings).pct_change().dropna()

        # Distribution plots
        fig = plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(returns, kde=True)
        plt.title(f'Portfolio {index + 1} Returns Distribution')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.subplot(1, 2, 2)
        stats.probplot(returns, dist="norm", plot=plt)
        plt.title(f'Portfolio {index + 1} Q-Q Plot')
        plt.savefig(distribution_image, bbox_inches='tight')
        plt.close()

        # Interactive distribution plots
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            f'Portfolio {index + 1} Returns Distribution', f'Portfolio {index + 1} Q-Q Plot'))
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
                          title_text=f"Portfolio {index + 1} Return Distribution Analysis")
        fig.write_html(interactive_dist)

        # Interactive dashboard
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            "Portfolio Performance", "Individual Holdings", "Returns Distribution", "Q-Q Plot"))
        fig.add_trace(go.Scatter(x=total_holdings.index, y=total_holdings.values, mode='lines', name='Portfolio'),
                      row=1, col=1)
        for crypto in cryptos:
            crypto_data = detailed_portfolio[detailed_portfolio['Crypto'] == crypto]
            fig.add_trace(go.Scatter(x=crypto_data['Date'], y=crypto_data['Holdings'], mode='lines', name=crypto),
                          row=1, col=2)
        fig.add_trace(go.Histogram(x=returns, nbinsx=30, name='Returns', histnorm='probability density'), row=2, col=1)
        fig.add_trace(go.Scatter(x=theoretical_quantiles, y=observed_quantiles, mode='markers', name='Q-Q Plot'), row=2,
                      col=2)
        fig.update_layout(height=1200, width=1200, title_text="Portfolio Analysis Dashboard", showlegend=True)
        fig.write_html(interactive_path)

        return True
    except Exception as e:
        logger.error(f"Error saving visualizations: {str(e)}")
        return False


def create_distribution_plots(returns, portfolio_index):
    fig = plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(returns, kde=True)
    plt.title(f'Portfolio {portfolio_index + 1} Returns Distribution')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.subplot(1, 2, 2)
    stats.probplot(returns, dist="norm", plot=plt)
    plt.title(f'Portfolio {portfolio_index + 1} Q-Q Plot')
    distribution_path = os.path.join(STATIC_DIR, f'portfolio_{portfolio_index}_distribution.png')
    plt.savefig(distribution_path, bbox_inches='tight')
    plt.close()
    return f'portfolio_{portfolio_index}_distribution.png'


def create_interactive_distribution_plots(returns, portfolio_index):
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
