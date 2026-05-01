# connection.py --> get_connection: Reads the file .env and takes: 
#   DB_HOST, DB_USER, DB_PASSWORD, etc.
#   also connects to MySQL, returns an active connection
from mdp.database.connection import get_connection

# Creates the constant that creates a table
CREATE_MARKET_CANDLES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS market_candles (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(32) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp_utc DATETIME NOT NULL,
    open_price DECIMAL(18, 8) NOT NULL,
    high_price DECIMAL(18, 8) NOT NULL,
    low_price DECIMAL(18, 8) NOT NULL,
    close_price DECIMAL(18, 8) NOT NULL,
    tick_volume BIGINT,
    spread INT,
    real_volume BIGINT,
    source VARCHAR(50) DEFAULT 'MT5',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT uq_market_candle UNIQUE (symbol, timeframe, timestamp_utc),     
    INDEX idx_symbol_timeframe_timestamp (symbol, timeframe, timestamp_utc)
);
"""

def create_tables() -> None:
    """
    Creates the core database tables required by the project.
    """
    connection = get_connection()

    # Sends Python instructions to MySQL
    cursor = connection.cursor()

    cursor.execute(CREATE_MARKET_CANDLES_TABLE_SQL)
    connection.commit()

    cursor.close()
    connection.close()