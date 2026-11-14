import os
from collections.abc import Generator
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine
from sqlalchemy.orm import Session, sessionmaker

# Locate the project root and load its .env file.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

# Validate required environment variables.
required_variables = [
    "POSTGRES_DB",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
]

missing_variables = [
    variable
    for variable in required_variables
    if not os.getenv(variable)
]

if missing_variables:
    raise RuntimeError(
        f"Missing environment variables: {', '.join(missing_variables)}. "
        "Check the .env file in the project root."
    )

# Build the database URL safely.
# URL.create handles special characters in passwords.
DATABASE_URL = URL.create(
    drivername="postgresql+psycopg",
    username=os.environ["POSTGRES_USER"],
    password=os.environ["POSTGRES_PASSWORD"],
    host="127.0.0.1",
    port=int(os.getenv("POSTGRES_PORT", "5433")),
    database=os.environ["POSTGRES_DB"],
)

# Create the SQLAlchemy engine.
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    echo=False,
)

# Session factory for database operations.
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def get_db() -> Generator[Session, None, None]:
    """Provide a database session and close it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
