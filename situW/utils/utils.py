import asyncio
import openai
import backoff
from retrying import retry

try:
    RateLimitError = openai.error.RateLimitError
    APIConnectionError = openai.error.APIConnectionError
except AttributeError:
    RateLimitError = openai.RateLimitError
    APIConnectionError = openai.APIConnectionError

@backoff.on_exception(backoff.expo, (RateLimitError, APIConnectionError))
def chat_completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

@backoff.on_exception(backoff.expo, (RateLimitError, APIConnectionError))
async def dispatch_openai_chat_requests(messages_list, model, temperature, max_tokens, top_p, stop_words):
    tasks = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            # max_completion_tokens=max_tokens,
            top_p=top_p,
            # stop=stop_words
        )
        for x in messages_list
    ]
    return await asyncio.gather(*tasks)

class OpenAIModel:
    def __init__(self, API_KEY, model_name, stop_words, max_new_tokens):
        openai.api_key = API_KEY
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words

    def chat_generate(self, input_string, max_token=None, temperature=1):
        response = chat_completions_with_backoff(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_string}
            ],
            max_completion_tokens=max_token if max_token else self.max_new_tokens,
            temperature=temperature
        )
        generated_text = response['choices'][0]['message']['content'].strip()
        finish_reason = response['choices'][0]['finish_reason']
        return generated_text, finish_reason

    def batch_chat_generate(self, messages_list, max_token=None, temperature=1):
        system_prompt = "You are a helpful assistant."
        open_ai_messages_list = [
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": msg}]
            for msg in messages_list
        ]
        max_new_token = max_token if max_token is not None else self.max_new_tokens
        
        responses = asyncio.run(
            dispatch_openai_chat_requests(
                open_ai_messages_list,
                self.model_name,
                temperature,
                max_new_token,
                1.0,
                self.stop_words
            )
        )
        # breakpoint()
        return [r['choices'][0]['message']['content'].strip() for r in responses]

    def batch_generate(self, messages_list, max_token=None, temperature=1):
        return self.batch_chat_generate(messages_list, max_token, temperature)

    def generate(self, input_string, max_token=None, temperature=1):
        return self.chat_generate(input_string, max_token, temperature)