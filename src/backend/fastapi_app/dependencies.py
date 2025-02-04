import logging
import os
from collections.abc import AsyncGenerator
from typing import Annotated, Optional, Union

from fastapi import Depends, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import asyncio

from visual_bge.modeling import Visualized_BGE

logger = logging.getLogger("ragapp")


class FastAPIAppContext(BaseModel):
    """
    Context for the FastAPI app
    """

    local_embed_dimensions: Optional[int]
    embedding_column: str


async def common_parameters():
    """
    Get the common parameters for the FastAPI app
    Use the pattern of `os.getenv("VAR_NAME") or "default_value"` to avoid empty string values
    """
    LOCAL_EMBED_HOST = os.getenv("LOCAL_EMBED_HOST")
    if LOCAL_EMBED_HOST == "imagebind":
        embedding_column = "img_embedding"
        local_embed_dimensions = 1024
    elif LOCAL_EMBED_HOST == "bge":
        embedding_column = "img_embedding_bge"
        local_embed_dimensions = 768
    else:
        #placeholder
        embedding_column = "img_embedding_bge"
        local_embed_dimensions = 768

    return FastAPIAppContext(
        local_embed_dimensions=local_embed_dimensions,
        embedding_column=embedding_column,
    )

async def create_async_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Get the agent database"""
    return async_sessionmaker(
        engine,
        expire_on_commit=False,
        autoflush=False,
    )


async def get_async_sessionmaker(
    request: Request,
) -> AsyncGenerator[async_sessionmaker[AsyncSession], None]:
    yield request.state.sessionmaker


async def get_context(
    request: Request,
) -> FastAPIAppContext:
    return request.state.context


async def get_async_db_session(
    sessionmaker: Annotated[async_sessionmaker[AsyncSession], Depends(get_async_sessionmaker)],
) -> AsyncGenerator[AsyncSession, None]:
    async with sessionmaker() as session:
        yield session

async def load_mm_embedding_model(
    # request: Request,
) -> Visualized_BGE:
    model = Visualized_BGE(model_name_bge="BAAI/bge-base-en-v1.5",
                    model_weight="/ssd_root/gao688/models/Visualized_base_en_v1.5.pth")
    model.eval()
    return model

async def get_mm_embedding_model(
    request: Request,
) -> Visualized_BGE:
    return request.state.embed_client

class ChatLLMClient(object):
    def __init__(self):
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        
        # Load configuration and adjust rope_scaling
        config = AutoConfig.from_pretrained(model_name)
        if hasattr(config, "rope_scaling"):
            config.rope_scaling = {"type": "linear", "factor": 1.0}  # Adjust as needed
        
        # Load model with the updated configuration
        self.model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=False)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        
    async def close(self):
        # del self.embedding_model
        del self.pipeline
        del self.model

async def initialize_chat_model_pipeline(
    # request: Request,
) -> ChatLLMClient:
    return ChatLLMClient()

async def get_local_chat_model(
    request: Request,
) -> ChatLLMClient:
    return request.state.chat_client


CommonDeps = Annotated[FastAPIAppContext, Depends(get_context)]
DBSession = Annotated[AsyncSession, Depends(get_async_db_session)]
# EmbeddingsModel = Annotated[SentenceTransformer, Depends(get_local_embedding_model)]
ImageEmbedModel = Annotated[Visualized_BGE, Depends(get_mm_embedding_model)]
LocalChatClient = Annotated[ChatLLMClient, Depends(get_local_chat_model)]
