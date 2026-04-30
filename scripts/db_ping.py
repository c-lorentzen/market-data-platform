from mdp.database.connection import get_connection


def main():
    connection = get_connection()

    if connection.is_connected():
        db_info = connection.server_info
        print("Database connection successful.")
        print(f"MySQL Server version: {db_info}")
        print(f"Connected database: {connection.database}")

    connection.close()
    print("Database connection closed.")


if __name__ == "__main__":
    main()