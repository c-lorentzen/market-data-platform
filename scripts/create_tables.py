from mdp.database.schema import create_tables


def main():
    create_tables()
    print("Database tables created successfully.")


if __name__ == "__main__":
    main()