from collections.abc import AsyncGenerator
from typing import Any, Final, Optional, Union, Dict

from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessageParam

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
from fastapi_app.query_rewriter import extract_search_arguments
from fastapi_app.rag_base import ChatParams, RAGChatBase
import asyncio

query_rewrite_prompt='''
You are a helpful assistant that help to rewrite user question for later queries over a product database. Your task is to extract the a search body for full-text search and some filters if exists.
Your output should be a JSON object, here is a detailed introduction of it. You can only return search_query if the input doesn't mention a filter explicitly.
Output Details:
- "search_query": Query string to use for full-text search.
- "stars_filter": Filters search results based on stars of the product.
    - `operator`: Operator to compare stars (`>`, `<`, `>=`, `<=`, `=`).
    - `value`: Value to compare stars against. Example: `50`.
- "price_filter": Filters search results based on price of the product.
    - `operator`: Operator to compare price (`>`, `<`, `>=`, `<=`, `=`).
    - `value`: Value to compare price against. Example: `50`.
- "is_best_seller_filter": Filters search whether the results are best sellers, the value can only be "True" or "False". 
    
For example:
1.
Input:
Find some best-selling red backpacks for me, they should be waterproof.
Output:
{
    "search_query": "waterproof red backpack",
    "is_best_seller_filter": "Ture",
}
2.
Input:
Are there any blue luggages less than $100 but having stars higher than 4.0?
Output:
{
    "search_query": "blue luggages",
    "price_filter": {
        "operator": "<",
        "value": "100"
        },
    "stars_filter": {
        "operator": ">",
        "value": "4.0"
        },
}

Now, given the following input provided by user, generate the corresponding output:
Input:
'''

class AdvancedRAGChat(RAGChatBase):
    def __init__(
        self,
        *,
        searcher: PostgresSearcher,
        chat_model: LocalChatClient,
    ):
        self.searcher = searcher
        self.chat_model = chat_model
        self.chat_token_limit = 1024

    
    async def generate_search_query(
        self,
        original_user_query: str,
        past_messages: list[ChatCompletionMessageParam],
    ) -> tuple[list[str], Union[Any, str, None], list]:

        msg = [
                {"role": "system", "content": query_rewrite_prompt},
                {"role": "user", "content": original_user_query},
        ]
        outputs = self.chat_model.pipeline(
            msg,
            max_new_tokens=self.chat_token_limit,
        )
        generated_content = outputs[0]["generated_text"][-1]['content']
        # print("=========after rewriting=========")
        # print(generated_content)
        query_text, filters = extract_search_arguments(generated_content)
        if query_text is None:
            query_text = original_user_query
        return [query_rewrite_prompt, original_user_query], query_text, filters
    
    async def prepare_context(
        self, chat_params: ChatParams
    ) -> tuple[list[str], list[Item], list[ThoughtStep]]:
        

        query_messages, query_text, filters = await self.generate_search_query(
            original_user_query=chat_params.original_user_query,
            past_messages=chat_params.past_messages,
        )
            
        # Retrieve relevant rows from the database with the GPT optimized query
        results = await self.searcher.search_and_embed(
            query_text,
            top=chat_params.top,
            enable_vector_search=chat_params.enable_vector_search,
            enable_text_search=chat_params.enable_text_search,
            filters=filters,
        )

        sources_content = [f"[{(item.asin)}]:{item.to_str_for_rag()}\n\n" for item in results]
        content = "\n".join(sources_content)
        
        prompt = {
            "instruction": chat_params.prompt_template,
            "hint": content,
        }

        thoughts = [
            ThoughtStep(
                title="Search using generated search arguments",
                description=query_text,
                props={
                    "top": chat_params.top,
                    "vector_search": chat_params.enable_vector_search,
                    "text_search": chat_params.enable_text_search,
                    "filters": filters,
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
        contextual_messages: list[ChatCompletionMessageParam],
        results: list[Item],
        earlier_thoughts: list[ThoughtStep],
    ) -> RetrievalResponse:
        chat_completion_response: ChatCompletion = await self.openai_chat_client.chat.completions.create(
            model=self.chat_deployment if self.chat_deployment else self.chat_model,
            messages=contextual_messages,
            temperature=chat_params.temperature,
            max_tokens=chat_params.response_token_limit,
            n=1,
            stream=False,
            seed=chat_params.seed,
        )

        return RetrievalResponse(
            message=Message(
                content=str(chat_completion_response.choices[0].message.content), role=AIChatRoles.ASSISTANT
            ),
            context=RAGContext(
                data_points={item.id: item.to_dict() for item in results},
                thoughts=earlier_thoughts
                + [
                    ThoughtStep(
                        title="Prompt to generate answer",
                        description=contextual_messages,
                        props=(
                            {"model": self.chat_model, "deployment": self.chat_deployment}
                            if self.chat_deployment
                            else {"model": self.chat_model}
                        ),
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
        if not self.chat_model:
            raise ValueError("Pipeline is not initialized. Call `initialize_pipeline()` first.")

        prompt = context['instruction'] + "\nFound Products:\n" + context['hint']
        
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
                + [],
            ),
        )
        async for chunk in stream_response():
            yield RetrievalResponseDelta(
                delta=Message(content=chunk, role=AIChatRoles.ASSISTANT)
            )

