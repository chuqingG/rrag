from typing import Optional, TypedDict
import asyncio

import torch
from visual_bge.modeling import Visualized_BGE

def encode_image_bge(img_path, model) -> list[float]:
    with torch.no_grad():
        img_emb = model.encode(image=img_path)
    return img_emb.tolist()[0]

def encode_query_bge(q, model) -> list[float]:
    with torch.no_grad():
        query_emb = model.encode(text=q)
    return query_emb.tolist()[0]

async def compute_text_embedding(
    q: str,
    embedding_model: Visualized_BGE,
    # embed_model: str = "all-MiniLM-L6-v2",
) -> list[float]:

    # Currently we set image model to cpu
    device = "cpu"
    embedding = await asyncio.to_thread(encode_query_bge, q, embedding_model)
    
    return embedding

