from pathlib import Path
from dotenv import load_dotenv
import os
import mysql.connector
from mysql.connector import Error


def load_environment() -> None:
    """
    Loads environment variables from the .env file located at the project root.
    """
    project_root = Path(__file__).resolve().parents[3]
    env_path = project_root / ".env"
    load_dotenv(env_path)


def get_connection():
    """
    Creates and returns a MySQL database connection using environment variables.
    """
    load_environment()

    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT", "3306")),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
        )

        if connection.is_connected():
            return connection

    except Error as error:
        raise RuntimeError(f"Database connection failed: {error}") from error