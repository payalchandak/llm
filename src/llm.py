import os
import litellm
from litellm import completion
from dotenv import load_dotenv
from openrouter import OpenRouter
import ipdb
load_dotenv()

import os
import difflib
import litellm
from litellm import completion
from openrouter import OpenRouter

class LLM: 
    def __init__(self, model_name: str):
        assert model_name is not None, "Model name must be specified"
        assert model_name in litellm.utils.get_valid_models(), "Model name must be a valid model"
        available_api_keys = [key for key in os.environ.keys() if key.__contains__("API_KEY")]
        print(available_api_keys)
        # if empty, raise error
        if len(available_api_keys) == 0:
            raise ValueError("No API keys found, must specify OpenRouter API key in environment variables")
        assert "OPENROUTER_API_KEY" in os.environ, "OpenRouter API key must be specified in environment variables"

        with OpenRouter(
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
        ) as open_router:

            list_of_models = [x.id for x in open_router.models.list().data]

        hms_azure_models = [
            "azure/gpt-5",
            "azure/gpt-5-mini",
            "azure/gpt-4.1",
            "azure/gpt-4.1-mini",
            "azure/gpt-4.1-nano",
        ]

        if model_name in hms_azure_models:
            self.model_name = f"azure/{model_name}"
        else:
            self.model_name = f"openrouter/{model_name}"


    def route_model(self, model_name: str):
        routed_model_name = model_name
        return routed_model_name

    def chat_completion(self, messages: list[dict], **kwargs):
        '''
        Accepts kwargs 
        def completion(
            model: str,
            messages: List = [],
            # Optional OpenAI params
            timeout: Optional[Union[float, int]] = None,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            n: Optional[int] = None,
            stream: Optional[bool] = None,
            stream_options: Optional[dict] = None,
            stop=None,
            max_completion_tokens: Optional[int] = None,
            max_tokens: Optional[int] = None,
            presence_penalty: Optional[float] = None,
            frequency_penalty: Optional[float] = None,
            logit_bias: Optional[dict] = None,
            user: Optional[str] = None,
            # openai v1.0+ new params
            response_format: Optional[dict] = None,
            seed: Optional[int] = None,
            tools: Optional[List] = None,
            tool_choice: Optional[str] = None,
            parallel_tool_calls: Optional[bool] = None,
            logprobs: Optional[bool] = None,
            top_logprobs: Optional[int] = None,
            safety_identifier: Optional[str] = None,
            deployment_id=None,
            # soon to be deprecated params by OpenAI
            functions: Optional[List] = None,
            function_call: Optional[str] = None,
            # set api_base, api_version, api_key
            base_url: Optional[str] = None,
            api_version: Optional[str] = None,
            api_key: Optional[str] = None,
            model_list: Optional[list] = None,  # pass in a list of api_base,keys, etc.
            # Optional liteLLM function params
            **kwargs,
        ) -> ModelResponse:
        Returns a chat completion response from the model in the format: 
        {
            'choices': [
                {
                'finish_reason': str,     # String: 'stop'
                'index': int,             # Integer: 0
                'message': {              # Dictionary [str, str]
                    'role': str,            # String: 'assistant'
                    'content': str          # String: "default message"
                }
                }
            ],
            'created': str,               # String: None
            'model': str,                 # String: None
            'usage': {                    # Dictionary [str, int]
                'prompt_tokens': int,       # Integer
                'completion_tokens': int,   # Integer
                'total_tokens': int         # Integer
            }
            }
        '''
        response = completion(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        return response

    def validate_response(self, response: dict):
        # check finish reason is valid 
        pass

    def get_response_content(self, response: dict):
        return response['choices'][0]['message']['content']

    def parse_response_to_structured_output(self, response: dict, schema: dict):
        # use free model to convert free text response to structured output
        pass

LLM("gpt-5")