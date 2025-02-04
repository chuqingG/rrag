import logging
from rich.logging import RichHandler
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TypedDict, Union
from dotenv import load_dotenv

import fastapi
from dotenv import load_dotenv

from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from visual_bge.modeling import Visualized_BGE

from fastapi_app.dependencies import (
    FastAPIAppContext,
    common_parameters,
    create_async_sessionmaker,
    ChatLLMClient,
    initialize_chat_model_pipeline,
    load_mm_embedding_model,
)
from fastapi_app.postgres_engine import create_postgres_engine_from_env

logger = logging.getLogger("ragapp")


class State(TypedDict):
    sessionmaker: async_sessionmaker[AsyncSession]
    context: FastAPIAppContext
    chat_client: ChatLLMClient
    embed_client: Visualized_BGE


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI) -> AsyncIterator[State]:
    context = await common_parameters()
    engine = await create_postgres_engine_from_env()
    sessionmaker = await create_async_sessionmaker(engine)
    chat_client = await initialize_chat_model_pipeline()
    embed_client = await load_mm_embedding_model()
    if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
        SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine)
    yield {"sessionmaker": sessionmaker, "context": context, "chat_client": chat_client, "embed_client": embed_client}
    await engine.dispose()
    
    logger.info("Closing embedding model...")
    del embed_client
    
    logger.info("Closing chat client...")
    await chat_client.close()

def create_app(testing: bool = False):
    load_dotenv()

    log_format = "%(message)s"
    console_handler = RichHandler(rich_tracebacks=True, show_time=False)
    file_handler = logging.FileHandler("rrag.log", mode="a")  # Logs to `app.log`
    file_formatter = logging.Formatter(log_format)  # No timestamp
    file_handler.setFormatter(file_formatter)
    
    logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[console_handler, file_handler]
        )

    app = fastapi.FastAPI(docs_url="/docs", lifespan=lifespan)

    from fastapi_app.routes import api_routes, frontend_routes

    app.include_router(api_routes.router)
    app.mount("/", frontend_routes.router)

    return app
