from collections.abc import AsyncGenerator
from typing import Optional, Union, Dict

from fastapi_app.api_models import (
    AIChatRoles,
    Message,
    RAGContext,
    RetrievalResponse,
    RetrievalResponseDelta,
    ThoughtStep,
)
from fastapi_app.dependencies import LocalChatClient
from fastapi_app.postgres_models import Item
from fastapi_app.postgres_searcher import PostgresSearcher
from fastapi_app.rag_base import ChatParams, RAGChatBase
import asyncio

fulltext_rewrite_prompt='''
You are a helpful assistant that help to rewrite user question for later queries over a product database. Your task is to extract the a search body for full-text search.
Remember to focus on the item itself. DO NOT include things like price, stars, whether bestsellers or not, etc.

For example:
1.
Input:
Find some best-selling red backpacks for me, they should be waterproof.
Output:
waterproof red backpack

2.
Input:
Are there any blue luggages less than $100 but having stars higher than 4.0?
Output:
blue luggages

Now, given the following input provided by user, generate the corresponding output:
Input:
'''

class SimpleRAGChat(RAGChatBase):
    def __init__(
        self,
        *,
        searcher: PostgresSearcher,
        chat_model: LocalChatClient,
    ):
        self.searcher = searcher
        self.chat_model = chat_model
        self.chat_token_limit = 1024

    
    async def prepare_context(
        self, chat_params: ChatParams
    ) -> tuple[list[str], list[Item], list[ThoughtStep]]:
        """Retrieve relevant rows from the database and build a context for the local model."""
        # Retrieve relevant rows from the database
        # chat_params.enable_text_search = False
        # query_messages, query_text, filters = await self.generate_search_query(
        #     original_user_query=chat_params.original_user_query,
        #     past_messages=chat_params.past_messages,
        # )
        msg = [
                {"role": "system", "content": fulltext_rewrite_prompt},
                {"role": "user", "content": chat_params.original_user_query},
        ]
        outputs = self.chat_model.pipeline(
            msg,
            max_new_tokens=self.chat_token_limit,
        )
        full_text_query = outputs[0]["generated_text"][-1]['content']
        print(full_text_query)
        
        results = await self.searcher.search_and_embed(
            full_text_query,
            top=chat_params.top,
            enable_vector_search=chat_params.enable_vector_search,
            enable_text_search=chat_params.enable_text_search,
        )

        sources_content = [f"[{(item.asin)}]:{item.to_str_for_rag()}\n\n" for item in results]
        content = "\n".join(sources_content)

        prompt = {
            "instruction": chat_params.prompt_template,
            "hint": content,
        }

        thoughts = [
            ThoughtStep(
                title="Search query for database",
                description=chat_params.original_user_query,
                props={
                    "top": chat_params.top,
                    "vector_search": chat_params.enable_vector_search,
                    "text_search": chat_params.enable_text_search,
                },
            ),
            ThoughtStep(
                title="Search results",
                description=[result.to_dict() for result in results],
            ),
        ]

        return prompt, results, thoughts
    
    async def answer(
        self,
        chat_params: ChatParams,
        contextual_messages: list[str],
        results: list[Item],
        earlier_thoughts: list[ThoughtStep],
    ) -> RetrievalResponse:
        """Generate a response using the local Hugging Face model."""
        if not self.chat_model:
            raise ValueError("chat model is not initialized. Call `initialize_pipeline()` first.")

        prompt = contextual_messages[0]

        # Use asyncio to run the pipeline's generate method in a thread
        async def generate_response():
    
            outputs = self.pipeline(
                prompt,
                max_new_tokens=chat_params.response_token_limit,
                temperature=chat_params.temperature,
                num_return_sequences=1,
            )
            return outputs[0]["generated_text"]

        generated_content = await asyncio.to_thread(generate_response)

        return RetrievalResponse(
            message=Message(content=generated_content, role=AIChatRoles.ASSISTANT),
            context=RAGContext(
                data_points={item.id: item.to_dict() for item in results},
                thoughts=earlier_thoughts
                + [
                    ThoughtStep(
                        title="Prompt to generate answer",
                        description=prompt,
                        props={"model": self.model_name},
                    ),
                ],
            ),
        )

    async def answer_stream(
        self,
        chat_params: ChatParams,
        context: Dict[str,str],
        results: list[Item],
        earlier_thoughts: list[ThoughtStep],
    ) -> AsyncGenerator[RetrievalResponseDelta, None]:
        """Simulate streaming by breaking the generated response into chunks."""
        if not self.chat_model:
            raise ValueError("Pipeline is not initialized. Call `initialize_pipeline()` first.")

        prompt = context['instruction'] + "\nFound Products:\n" + context['hint']
        print("=======================================================")
        print(prompt)
        async def stream_response():
            msg = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": chat_params.original_user_query},
            ]
            outputs = self.chat_model.pipeline(
                msg,
                max_new_tokens=chat_params.response_token_limit,
            )
            generated_content = outputs[0]["generated_text"][-1]['content']
            for chunk in generated_content.split(" "):  # Stream word-by-word
                yield chunk+" "

        yield RetrievalResponseDelta(
            context=RAGContext(
                data_points={item.asin: item.to_dict() for item in results},
                thoughts=earlier_thoughts
                + [
                    ThoughtStep(
                        title="Prompt to generate answer",
                        description=prompt,
                        props={"model": "local llama3"},
                    ),
                ],
            ),
        )

        async for chunk in stream_response():
            yield RetrievalResponseDelta(
                delta=Message(content=chunk, role=AIChatRoles.ASSISTANT)
            )
