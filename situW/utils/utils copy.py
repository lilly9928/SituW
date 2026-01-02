import backoff  # for exponential backoff
import openai
import os
import asyncio
from typing import Any
import random
from retrying import retry
from openai import AsyncOpenAI


try:
    # openai 1.x
    RateLimitError = openai.RateLimitError
    APIConnectionError = openai.APIConnectionError
except AttributeError:
    # openai 0.x
    RateLimitError = openai.error.RateLimitError
    APIConnectionError = openai.error.APIConnectionError

@backoff.on_exception(backoff.expo, (RateLimitError, APIConnectionError))
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

@backoff.on_exception(backoff.expo, (RateLimitError, APIConnectionError))
def chat_completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


async def dispatch_openai_chat_requests(
    client: AsyncOpenAI,
    messages_list,
    model: str,
    temperature: float,
    max_tokens: int,
    stop_words=None,
):
    tasks = [
        client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,      # chat.completions에서는 보통 max_tokens
            stop=stop_words,
        )
        for msgs in messages_list
    ]
    return await asyncio.gather(*tasks)



# async def dispatch_openai_chat_requests(
#     messages_list: list[list[dict[str,Any]]],
#     model: str,
#     temperature: float,
#     max_tokens: int,
#     top_p: float,
#     stop_words: list[str]
# ) -> list[str]:
#     """Dispatches requests to OpenAI API asynchronously.
    
#     Args:
#         messages_list: List of messages to be sent to OpenAI ChatCompletion API.
#         model: OpenAI model to use.
#         temperature: Temperature to use for the model.
#         max_tokens: Maximum number of tokens to generate.
#         top_p: Top p to use for the model.
#         stop_words: List of words to stop the model from generating.
#     Returns:
#         List of responses from OpenAI API.
#     """
#     breakpoint()
#     async_responses = [
#         openai.ChatCompletion.acreate(
#             model=model,
#             messages=x,
#             temperature=temperature,
#             max_tokens=max_tokens,
#             top_p=top_p,
#             # stop = stop_words
#         )
#         for x in messages_list
#     ]
#     breakpoint()
#     return await asyncio.gather(*async_responses)

async def dispatch_openai_prompt_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str]
) -> list[str]:
    async_responses = [
        openai.Completion.acreate(
            model=model,
            prompt=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty = 0.0,
            presence_penalty = 0.0,
            stop = stop_words
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)

class OpenAIModel:
    def __init__(self, API_KEY, model_name, stop_words, max_new_tokens) -> None:
        openai.api_key = API_KEY
        self.async_client = AsyncOpenAI(api_key=API_KEY)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words
        
    async def batch_chat_generate_async(self, messages_list, max_token=None, temperature=0.0):
        system_prompt = "You are a helpful assistant."
        open_ai_messages_list = [
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": message}]
            for message in messages_list
        ]

        max_new_token = max_token if max_token is not None else self.max_new_tokens

        resps = await dispatch_openai_chat_requests(
            client=self.async_client,
            messages_list=open_ai_messages_list,
            model=self.model_name,
            temperature=temperature,
            max_tokens=max_new_token,
            stop_words=self.stop_words,
        )

        texts = [r.choices[0].message.content.strip() for r in resps]
        finish_reasons = [r.choices[0].finish_reason for r in resps]
        return texts, finish_reasons
    
    
    # @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def chat_generate(self, input_string, max_token, temperature = 0.0):
        # breakpoint()
        response = chat_completions_with_backoff(
                model = self.model_name,
                messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": input_string}
                    ],
                # temperature = temperature,
                # top_p = 1,
                # stop = self.stop_words
        )
        generated_text = response['choices'][0]['message']['content'].strip()
        finish_reason = response['choices'][0]['finish_reason']
        return generated_text, finish_reason
    
    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def prompt_generate(self, input_string, temperature = 0.0):
        response = completions_with_backoff(
            model = self.model_name,
            prompt = input_string,
            max_tokens = self.max_new_tokens,
            temperature = temperature,
            top_p = 1.0,
            frequency_penalty = 0.0,
            presence_penalty = 0.0,
            stop = self.stop_words
        )
        generated_text = response['choices'][0]['text'].strip()
        return generated_text

    def generate(self, input_string, max_token=None, temperature = 0.0):
        if True:
            return self.chat_generate(input_string, max_token, temperature)
        else:
            raise Exception("Model name not recognized")
    
    def batch_chat_generate(self, messages_list, max_token=None, temperature = 0.0):
        open_ai_messages_list = []
        system_prompt = "You are a helpful assistant."
        for message in messages_list:
            open_ai_messages_list.append(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]
            )
        breakpoint()
        max_new_token = self.max_new_tokens
        predictions = asyncio.run(
            dispatch_openai_chat_requests(
                    open_ai_messages_list, self.model_name, temperature, max_new_token, 1.0, self.stop_words
            )
        )
        breakpoint()
        finish_reason = [x['choices'][0]['finish_reason'].strip() for x in predictions]
        return [x['choices'][0]['message']['content'].strip() for x in predictions]
    
    def batch_prompt_generate(self, prompt_list, temperature = 0.0):
        predictions = asyncio.run(
            dispatch_openai_prompt_requests(
                    prompt_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words
            )
        )
        return [x['choices'][0]['text'].strip() for x in predictions]

    def batch_generate(self, messages_list, max_token=None, temperature = 0.0):
        if True:
            breakpoint()
            return self.batch_chat_generate(messages_list, max_token, temperature)
        else:
            raise Exception("Model name not recognized")

    def generate_insertion(self, input_string, suffix, temperature = 0.0):
        response = completions_with_backoff(
            model = self.model_name,
            prompt = input_string,
            suffix= suffix,
            temperature = temperature,
            max_tokens = self.max_new_tokens,
            top_p = 1.0,
            frequency_penalty = 0.0,
            presence_penalty = 0.0
        )
        generated_text = response['choices'][0]['text'].strip()
        return generated_text