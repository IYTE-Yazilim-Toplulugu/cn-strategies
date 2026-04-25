"""
Bu dosya tüm ortak ayarları tutar.

Buradaki amaç:
- parametreleri tek yerden yönetmek
- stratejiler arasında adil kıyas yapmak
- Safe / Fast / ML stratejilerinin aynı omurgayı kullanmasını sağlamak
"""

# Başlangıç sermayesi
INITIAL_CAPITAL = 3000.0

# --- Safe Strategy için MA ayarları ---
SHORT_MA = 20
LONG_MA = 50
USE_EMA = False  # Safe stratejide varsayılan olarak SMA kullanacağız

# --- Fast Strategy için EMA ayarları ---
FAST_EMA_SHORT = 9
FAST_EMA_LONG = 21

# Bollinger Band ayarları
BB_WINDOW = 20
BB_STD = 2

# Hacim filtresi
VOLUME_AVG_WINDOW = 20
VOLUME_SPIKE_MULTIPLIER = 1.5

# Volatilite guard ayarları
VOLATILITY_WINDOW = 20
VOLATILITY_GUARD_MULTIPLIER = 1.8

# İşlem ayarları
DEFAULT_ALLOCATION = 0.15
STRONG_SIGNAL_ALLOCATION = 0.20
FAST_ALLOCATION = 0.15
ML_ALLOCATION = 0.15

DEFAULT_LEVERAGE = 2 # Reduced from 3 to 2 for safety considering compound returns and ML max drawdown

# Risk / reward ayarları
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 3.0

STOP_LOSS_PCT = 0.03
RR_RATIO = 2.5
TAKE_PROFIT_PCT = STOP_LOSS_PCT * RR_RATIO

FAST_STOP_LOSS_PCT = 0.025
FAST_TAKE_PROFIT_PCT = FAST_STOP_LOSS_PCT * RR_RATIO

ML_STOP_LOSS_PCT = 0.03
ML_TAKE_PROFIT_PCT = ML_STOP_LOSS_PCT * RR_RATIO

# Banda çok yapışıkken giriş yapmamak için tamponlar
UPPER_BAND_BUFFER = 0.997
LOWER_BAND_BUFFER = 1.003

RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Gerekli minimum geçmiş veri
MIN_HISTORY = 80

SAFE_MIN_MA_GAP_RATIO = 0.0005
SAFE_REQUIRE_BB_MID_CONFIRM = True
SAFE_REQUIRE_VOLUME_SPIKE = False
SAFE_STRONG_TREND_GAP_RATIO = 0.0012

ML_PROBA_THRESHOLD = 0.55

# --- Walk-forward split ayarları ---
TRAIN_RATIO = 0.75   # ilk %75 geliştirme, son %25 pseudo unseen test

# --- Block bootstrap ayarları ---
BOOTSTRAP_BLOCK_MIN = 5
BOOTSTRAP_BLOCK_MAX = 20
BOOTSTRAP_SEED = 42
BOOTSTRAP_SCENARIOS = 10