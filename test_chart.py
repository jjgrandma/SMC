import sys, os
os.environ['MPLBACKEND'] = 'Agg'
sys.path.insert(0, '.')
os.environ['OPENAI_API_KEY'] = 'sk-test'
os.environ['TELEGRAM_BOT_TOKEN'] = 'test:token'

from app.tools import get_market_data, get_current_price
from app.chart import chart_mtf, chart_signal

print('Fetching data...')
df_h1 = get_market_data('XAUUSDm', 'H1')
df_h4 = get_market_data('XAUUSDm', 'H4')
df_d1 = get_market_data('XAUUSDm', 'D1')
price = get_current_price('XAUUSDm')
print('Price:', price['mid'])

print('MTF chart...')
buf = chart_mtf(df_d1, df_h4, df_h1, 'XAUUSDm', price['mid'])
kb = len(buf.read()) // 1024
print('MTF chart:', kb, 'KB - OK')

print('Signal chart...')
sig = {
    'action': 'BUY', 'entry': 4600.0, 'stop_loss': 4580.0,
    'take_profit': 4650.0, 'rr_ratio': 2.5, 'confidence': 'HIGH',
    'confluences': ['BOS bullish H1', 'OB at 4595'],
    'current_price': {'mid': price['mid']},
}
buf2 = chart_signal(df_h1, 'XAUUSDm', 'H1', sig)
kb2 = len(buf2.read()) // 1024
print('Signal chart:', kb2, 'KB - OK')
print('ALL CHARTS OK')
