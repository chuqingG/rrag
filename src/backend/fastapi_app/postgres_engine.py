import logging
import os

from pgvector.asyncpg import register_vector
from sqlalchemy import event
from sqlalchemy.engine import AdaptedConnection
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

logger = logging.getLogger("ragapp")


async def create_postgres_engine(*, host, username, database, password, sslmode) -> AsyncEngine:
    logger.info("Authenticating to PostgreSQL using password...")

    # Chuqing: change this for my pg@16
    DATABASE_URI = f"postgresql+asyncpg://{username}:{password}@{host}:5433/{database}"
    # Specify SSL mode if needed
    if sslmode:
        DATABASE_URI += f"?ssl={sslmode}"

    engine = create_async_engine(DATABASE_URI, echo=False)

    @event.listens_for(engine.sync_engine, "connect")
    def register_custom_types(dbapi_connection: AdaptedConnection, *args):
        logger.info("Registering pgvector extension...")
        try:
            dbapi_connection.run_async(register_vector)
        except ValueError:
            logger.warning("Could not register pgvector data type yet as vector extension has not been CREATED")

    return engine


async def create_postgres_engine_from_env() -> AsyncEngine:

    return await create_postgres_engine(
        host=os.environ["POSTGRES_HOST"],
        username=os.environ["POSTGRES_USERNAME"],
        database=os.environ["POSTGRES_DATABASE"],
        password=os.environ.get("POSTGRES_PASSWORD"),
        sslmode=os.environ.get("POSTGRES_SSL"),
    )


async def create_postgres_engine_from_args(args) -> AsyncEngine:

    return await create_postgres_engine(
        host=args.host,
        username=args.username,
        database=args.database,
        password=args.password,
        sslmode=args.sslmode,
    )
