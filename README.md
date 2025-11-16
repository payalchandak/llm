# LLM Interface

A unified interface for querying Large Language Models (LLMs) across multiple providers using LiteLLM and OpenRouter. This package provides intelligent model routing that automatically selects the best provider for each model request.

## What is LiteLLM?

[LiteLLM](https://github.com/BerriAI/litellm) is a Python library that provides a unified interface to call multiple LLM APIs with a consistent OpenAI-like API. It supports 100+ LLM providers including:

- **OpenAI** (GPT-4, GPT-3.5, etc.)
- **Anthropic** (Claude models)
- **Azure OpenAI**
- **Google** (Gemini, PaLM)
- **OpenRouter** (aggregator for multiple models)
- And many more...

LiteLLM handles provider-specific differences, retries, rate limiting, and error handling, allowing you to switch between providers with minimal code changes.

## What is OpenRouter?

[OpenRouter](https://openrouter.ai/) is a unified API that provides access to 100+ LLM models from various providers through a single interface. It's particularly useful when:

- You don't have direct API keys for specific providers
- You want to access models not available through your direct provider accounts
- You need a fallback option when your primary provider is unavailable
- You want to compare models across different providers

OpenRouter requires credits (free tier available) and routes requests to the appropriate provider on your behalf.

## Architecture

This package uses a three-tier routing system:

1. **Azure** (highest priority): Direct Azure OpenAI deployments
2. **Provider** (medium priority): Direct API access (OpenAI, Anthropic, etc.)
3. **OpenRouter** (fallback): Unified API for models not available through other routes

The routing is handled by a "routing judge" - an LLM that intelligently selects the best route based on:
- Model name matching (semantic and exact)
- Available API keys
- Model availability in each catalog

## Setup

### 1. Install Dependencies

```bash
pip install litellm openrouter python-dotenv pydantic
```

### 2. Environment Variables

Create a `.env` file in your project root with the following API keys:

#### Required (at least one)

- **`OPENROUTER_API_KEY`**: Required if using OpenRouter models or as the routing judge. Get your key from [OpenRouter](https://openrouter.ai/keys).

#### Optional (for direct provider access)

- **`OPENAI_API_KEY`**: For direct OpenAI API access
- **`ANTHROPIC_API_KEY`**: For direct Anthropic/Claude API access
- **`GOOGLE_API_KEY`**: For direct Google/Gemini API access
- Any other provider API keys following the pattern `{PROVIDER}_API_KEY`

#### Optional (for Azure)

- **`AZURE_API_KEY`**: Azure OpenAI API key
- **`AZURE_API_BASE`**: Azure OpenAI endpoint URL
- **`AZURE_API_VERSION`**: Azure API version
- **`AZURE_API_MODELS`**: Comma-separated list of available Azure models (e.g., `"gpt-5,gpt-4.1,gpt-4.1-mini"`)

### 3. Example `.env` File

```env
# OpenRouter (required for routing judge and fallback models)
OPENROUTER_API_KEY=sk-or-v1-...

# Direct provider access (optional but recommended)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Azure (optional)
AZURE_API_KEY=...
AZURE_API_BASE=https://your-resource.openai.azure.com/
AZURE_API_VERSION=2024-02-15-preview
AZURE_API_MODELS=gpt-5,gpt-4.1,gpt-4.1-mini
```

## Usage

### Basic Example

```python
from llm import LLM

# Initialize an LLM - routing happens automatically
llm = LLM("gpt-4o")

# Make a completion request
response = llm.completion(
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    temperature=0.7,
    max_tokens=100
)

print(response.choices[0].message.content)
```

### Advanced Example with Structured Output

```python
from llm import LLM

llm = LLM("claude-sonnet-4.5")

response = llm.completion(
    messages=[{"role": "user", "content": "Extract the key points from this text..."}],
    temperature=0.3,
    max_tokens=500,
    response_format={"type": "json_object"}
)
```

### Custom Routing Judge

By default, the system uses `openrouter/openai/gpt-4o-mini` as the routing judge. You can specify a different model:

```python
llm = LLM("gpt-5-2025-11-16", routing_judge="azure/gpt-4.1-mini")
```

## What the Code Does

### `LLM` Class

The `LLM` class is a thin wrapper that:

1. **Resolves the model**: Takes a user-friendly model name (e.g., `"gpt-5-2025-11-16"`) and resolves it to a concrete provider-specific model ID (e.g., `"azure/gpt-5"` or `"openrouter/openai/gpt-3.5-turbo"`)

2. **Tests the connection**: On initialization, sends a test request to verify the model is accessible and working

3. **Exposes a simple API**: Provides a `completion()` method that wraps `litellm.completion()` with the resolved model

### `ModelRouter` Class

The `ModelRouter` class handles intelligent routing:

1. **Loads model catalogs**: 
   - Azure models from `AZURE_API_MODELS` environment variable
   - Provider models from LiteLLM's catalog (based on available API keys)
   - OpenRouter models by querying the OpenRouter API

2. **Exact matching**: First tries to find exact matches in the catalogs (prioritizing Azure → Provider → OpenRouter)

3. **LLM-based routing**: If no exact match, uses a "routing judge" LLM to decide which route to use

4. **Model resolution**: Uses the routing judge again to map the requested model name to a specific model in the selected route

5. **Fallback handling**: If the selected route has no available models, falls back to other routes

## Example Behavior

When you initialize an LLM, you'll see output like this:

```
Routing model gpt-5-2025-11-16 to valid LLM...
Selected route azure because the requested model 'gpt-5-2025-11-16' semantically matches the azure model 'gpt-5'.
Resolved gpt-5-2025-11-16 to azure/gpt-5
Testing LLM at azure/gpt-5
Successfully recieved response from gpt-5-2025-08-07
```

### Routing Examples

**Azure Route** (when model matches Azure deployment):
```
Routing model gpt-5-2025-11-16 to valid LLM...
Selected route azure because the requested model 'gpt-5-2025-11-16' semantically matches the azure model 'gpt-5'.
Resolved gpt-5-2025-11-16 to azure/gpt-5
```

**Provider Route** (when direct API key is available):
```
Routing model claude-sonnet-4.5 to valid LLM...
Selected route provider because the requested model 'claude-sonnet-4.5' matches a model available from the provider with a direct api key.
Resolved claude-sonnet-4.5 to claude-sonnet-4-5
```

**OpenRouter Route** (fallback when no direct access):
```
Routing model gpt-3.5-turbo to valid LLM...
Selected route openrouter because requested model 'gpt-3.5-turbo' does not match any available azure models and there are no applicable provider options.
Resolved gpt-3.5-turbo to openrouter/openai/gpt-3.5-turbo
```

### Error Handling

The system validates each model on initialization. If a model fails, you'll see an error:

```
RuntimeError: Could not get a valid response from openrouter/deepseek/deepseek-r1-0528. 
litellm.APIError: APIError: OpenrouterException - {"error":{"message":"This request requires more credits, 
or fewer max_tokens. You requested up to 7168 tokens, but can only afford 4706..."}}
```

Common issues:
- **Insufficient credits**: OpenRouter account needs more credits
- **Invalid API key**: Check your environment variables
- **Model unavailable**: The requested model may not be available on the selected route
- **Rate limiting**: Provider may be rate limiting requests

## Features

- ✅ **Unified API**: Same interface for all LLM providers
- ✅ **Intelligent Routing**: Automatically selects the best provider
- ✅ **Fallback Support**: Gracefully falls back to OpenRouter when needed
- ✅ **Connection Testing**: Validates models on initialization
- ✅ **Flexible Configuration**: Supports Azure, direct providers, and OpenRouter
- ✅ **Full LiteLLM Support**: All LiteLLM parameters and features available

## Files

- **`src/llm.py`**: Main `LLM` class wrapper
- **`src/model_router.py`**: Intelligent routing logic
- **`src/answers.py`**: Structured output schemas (if used)
- **`src/trylitellm.py`**: Example usage and testing

## Notes

- The routing judge defaults to `openrouter/openai/gpt-4o-mini` but can be customized
- Model names are normalized (lowercase, whitespace removed) for matching
- The system prioritizes Azure → Provider → OpenRouter routes
- All LiteLLM features (streaming, tools, structured output, etc.) are supported through the `completion()` method's `**kwargs`
