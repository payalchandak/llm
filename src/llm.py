from __future__ import annotations
import json
import os
import re
import difflib
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Literal, Optional
import litellm
from litellm import completion
from dotenv import load_dotenv
from openrouter import OpenRouter
from pydantic import BaseModel, StringConstraints
from typing import Annotated
import ipdb
load_dotenv()
litellm.enable_json_schema_validation = True

# litellm._turn_on_debug()

log = logging.getLogger(__name__)

class RouteDecision(BaseModel):
    route: Annotated[str, StringConstraints(pattern=r"^(azure|provider|openrouter)$")]
    reason: str

class RouteDecider:
    """
    Ask a small LLM to choose route only.
    It returns a JSON object with route and reason.
    We enforce that the final id still passes the safety gate.
    """

    def __init__(
        self,
        judge_model: str = "openrouter/openai/gpt-4o-mini",
    ):
        self.judge_model = judge_model
        if "OPENROUTER_API_KEY" not in os.environ:
            raise RuntimeError("OPENROUTER_API_KEY must be set")

    def decide(
        self,
        requested_model: str,
        azure_models: list,
        provider_models: list,
    ) -> RouteDecision:
        facts = {
            "requested_model": requested_model,
            "azure_models_available": azure_models,
            "provider_api_keys_available": [k for k in os.environ.keys() if 'API_KEY' in k and k not in ["OPENROUTER_API_KEY", "AZURE_API_KEY"]],
            "provider_models_available": provider_models,
        }
        messages = self._build_prompt(facts)
        response = completion(
            model=self.judge_model,
            messages=messages,
            response_format=RouteDecision,
        )
        return RouteDecision.model_validate_json(response.choices[0].message.content)

    @staticmethod
    def _build_prompt(facts: dict) -> list[dict]:
        sys = (
            "You are a routing judge. Choose the single best route for a model request.\n"
            "Rules:\n"
            "1. If the requested model is in azure_allowlist then you must prefer azure. Allow the requested model name to be a semantic match of the azure model name (e.g. 'gpt-5-2025-11-16' matches 'gpt-5'), but do not allow cross-version matches (e.g. 'gpt-4o' does not match 'gpt-4.1').\n"
            "2. Else if requested model matches a provider for which a direct API key is available then you must choose provider.\n" 
            "For example, if requested model is 'gpt-4o' and an API key for OpenAI is available then you must choose provider. Or if requested model is 'gemini-flash' and an API key for Google is available then you must choose provider.\n" 
            "You can verify the availability of the provider models by checking the provider_models_available list."
            "3. Else choose openrouter. Your default choice should be openrouter.\n"
            "Return strict JSON with keys route and reason. Route must be one of azure, provider, openrouter.\n"
            "No extra text."
        )
        return [{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(facts, ensure_ascii=False)}]

class ModelResolver:
    """
    Orchestrates judge route selection and safety gated id resolution.
    """

    def __init__(self):
        if not os.getenv("OPENROUTER_API_KEY", ""):
            raise RuntimeError("OPENROUTER_API_KEY must be set in environment")

    def resolve(self, requested_model: str) -> str:
        if not requested_model or not isinstance(requested_model, str):
            raise RuntimeError("model name must be a non empty string")

        azure_models = os.getenv("AZURE_API_MODELS", "").split(',')
        provider_models = [x for x in litellm.utils.get_valid_models() if 'openrouter/' not in x.lower() and 'azure' not in x.lower()]
        openrouter_models = [x.id for x in OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY")).models.list().data]

        judge = RouteDecider()
        decision = judge.decide(
            requested_model=requested_model,
            azure_models=azure_models,
            provider_models=provider_models,
        )

        if decision.route == "azure":
            available_models = azure_models
        elif decision.route == "provider":
            available_models = provider_models
        else:
            available_models = openrouter_models

        print(decision.route)

        prompt =  f"""
        You are a model resolver. 
        The requested model is {requested_model}.
        The requested model {requested_model} will be provided through {decision.route} because {decision.reason}.
        You are given a list of available models. 
        You need to resolve the model to one of the available models.
        Respond exactly with one of the available model options and nothing else
        """

        response = completion(
            model="openrouter/openai/gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(available_models, ensure_ascii=False)}],
        )

        resolved_model = response.choices[0].message.content
        assert resolved_model in available_models, f"Resolved model {resolved_model} is not in the available models {available_models}"
        if decision.route == "azure":
            resolved_model = f"azure/{resolved_model}"
        elif decision.route == "openrouter":
            resolved_model = f"openrouter/{resolved_model}"

        return resolved_model

# ---------- Public resolver and LLM wrapper ----------
# payal: remove this dataclass 
@dataclass(frozen=True)
class ResolverConfig:
    azure_allowlist: set[str] = frozenset({
        "gpt-5",
        "gpt-5-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
    })
    # safety policy defaults
    allow_date_suffix: bool = True
    allow_latest_alias: bool = True
    allow_cross_family: bool = False
    allow_tier_change: bool = False


class LLM:
    """
    Thin wrapper that resolves a model once and exposes the chosen id.
    """

    def __init__(self, model_name: str):
        self._resolver = ModelResolver()
        self.model_name = self._resolver.resolve(model_name)
        print('resolved model:', self.model_name)
        # Optional: attach why route was chosen by letting ModelResolver expose it if desired
        # For brevity, omitted here. You can capture and store judge.reason if useful.


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

LLM("gpt-5-2025-11-16")
LLM("claude-sonnet-4.5")   
LLM("gemini-flash")