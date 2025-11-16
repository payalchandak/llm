from __future__ import annotations

from dotenv import load_dotenv
import litellm
from litellm import completion
from model_router import ModelRouter

# Load environment and enable json schema validation once
load_dotenv()
litellm.enable_json_schema_validation = True

class LLM:
    """Thin wrapper that resolves a model once and exposes the chosen id."""

    def __init__(self, model_name: str, routing_judge: str = "openrouter/openai/gpt-4o-mini"):
        print(f"Routing model {model_name} to valid LLM...")
        self._router = ModelRouter(routing_judge=routing_judge)
        self.model_name = self._router.resolve(model_name)
        print(f"Resolved {model_name} to {self.model_name}")
        print(f"Testing LLM at {self.model_name}")
        try:
            response = self.completion([{"role": "user", "content": "Hello, model! Can you reply with 'hi'?"},])
            if response.choices[0].finish_reason != "stop":
                raise RuntimeError(f"The finish reason in the model response was not 'stop': {response}")
            print(f"Successfully recieved response from {response.model}")
        except Exception as e:
            error_msg = str(e).split('\n')[0] if '\n' in str(e) else str(e)
            raise RuntimeError(f"Could not get a valid response from {self.model_name}. {error_msg}") from None

    def completion(self, messages: list[dict], **kwargs):
        """Generate a chat completion using the resolved model.
        
        This method wraps litellm.completion() and uses the model resolved during
        LLM initialization. All standard litellm/OpenAI completion parameters are
        supported via **kwargs.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                Example: [{"role": "user", "content": "Hello!"}]
            **kwargs: Additional parameters passed to litellm.completion(). Common options:
                
                **Sampling Parameters:**
                - temperature (float): Sampling temperature (0-2). Higher = more random.
                - top_p (float): Nucleus sampling parameter.
                - seed (int): Random seed for deterministic outputs.
                
                **Generation Control:**
                - max_tokens (int): Maximum tokens to generate.
                - max_completion_tokens (int): Alternative max tokens parameter.
                - stop (str or list): Stop sequences.
                - n (int): Number of completions to generate.
                
                **Streaming:**
                - stream (bool): If True, stream partial progress.
                - stream_options (dict): Options for streaming responses.
                
                **Penalties:**
                - presence_penalty (float): Penalize tokens based on presence.
                - frequency_penalty (float): Penalize tokens based on frequency.
                - logit_bias (dict): Modify likelihood of specific tokens.
                
                **Structured Output:**
                - response_format (dict): Force JSON or other structured output.
                - tools (list): List of tool/function definitions.
                - tool_choice (str or dict): Control tool usage.
                - parallel_tool_calls (bool): Allow parallel tool calls.
                
                **Logging & Debugging:**
                - logprobs (bool): Include log probabilities.
                - top_logprobs (int): Number of top logprobs to return.
                
                **API Configuration:**
                - timeout (float or int): Request timeout in seconds.
                - base_url (str): Custom API base URL.
                - api_version (str): API version.
                - api_key (str): Override API key.
                - model_list (list): List of API configurations.
                - deployment_id (str): Deployment ID for Azure.
                - user (str): Unique identifier for the end-user.
                
                **Deprecated (OpenAI):**
                - functions (list): Deprecated, use 'tools' instead.
                - function_call (str): Deprecated, use 'tool_choice' instead.
        
        Returns:
            ModelResponse: A response object with the following structure:
                - choices (list): List of completion choices, each containing:
                    - finish_reason (str): Reason for completion ('stop', 'length', etc.)
                    - index (int): Index of the choice
                    - message (dict): Message object with:
                        - role (str): Message role ('assistant', etc.)
                        - content (str): Generated text content
                        - tool_calls (list, optional): Tool/function calls if used
                - model (str): Model identifier used for the completion
                - created (int): Unix timestamp of creation
                - usage (dict): Token usage statistics:
                    - prompt_tokens (int): Tokens in the prompt
                    - completion_tokens (int): Tokens in the completion
                    - total_tokens (int): Total tokens used
        
        Example:
            >>> llm = LLM("gpt-4o")
            >>> response = llm.completion(
            ...     messages=[{"role": "user", "content": "Hello!"}],
            ...     temperature=0.7,
            ...     max_tokens=100
            ... )
            >>> print(response.choices[0].message.content)
        """
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