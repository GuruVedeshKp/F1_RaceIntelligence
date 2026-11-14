from sqlalchemy import text

from f1ri.db.session import engine


def main() -> None:
    """Check the PostgreSQL connection."""

    try:
        with engine.connect() as connection:
            result = connection.execute(
                text(
                    """
                    SELECT
                        current_database(),
                        current_user,
                        version();
                    """
                )
            )

            database, user, version = result.one()

        print("Database connection successful!")
        print(f"Database: {database}")
        print(f"User: {user}")
        print(f"Server: {version.splitlines()[0]}")

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
