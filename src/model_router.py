from __future__ import annotations

import json
import os
import time
from typing import List
import difflib
import litellm
from litellm import completion
from openrouter import OpenRouter
from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    route: str = Field(pattern=r"^(azure|provider|openrouter)$")
    reason: str


class ModelRouter:
    """Orchestrates route selection and model id resolution."""

    def __init__(self, routing_judge: str = "openrouter/openai/gpt-4o-mini", request_timeout: int = 60, max_prompt_items: int = 200):
        self.routing_judge = routing_judge
        self.request_timeout = int(request_timeout)
        self.max_prompt_items = int(max_prompt_items)

        # Judge defaults to an OpenRouter path, so this key is typically required.
        # If you point routing_judge to a direct provider, set the right env for that provider.
        if self.routing_judge.startswith("openrouter/") and not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY must be set when using an OpenRouter routing judge")

    # ---- Public API ----
    def resolve(self, requested_model: str) -> str:
        """Resolve a user requested model into a concrete provider specific model id.

        Returns a string like:
            - "azure/gpt-4.1"
            - "openrouter/openai/gpt-4o-mini"
            - "gpt-4o" (for direct provider routes)
        """
        if not isinstance(requested_model, str) or not requested_model.strip():
            raise ValueError("model name must be a non empty string")

        requested_model = self._normalize(requested_model)
        azure_models, provider_models, openrouter_models = self._load_catalogs()

        # Fast path exact hits across catalogs, honoring priority: azure then provider then openrouter
        exact = self._find_exact(requested_model, azure_models, provider_models, openrouter_models)
        if exact:
            route, model = exact
            return self._prefix(route, model)

        # Ask judge to choose route
        decision = self._decide_route(
            requested_model=requested_model,
            azure_models=azure_models,
            provider_models=provider_models,
        )
        print(f"Selected route {decision.route} because {decision.reason.lower()}")

        available = self._pick_available(decision.route, azure_models, provider_models, openrouter_models)
        if not available:
            # Fallback chain if judge picked an empty route
            for r in ("provider", "openrouter", "azure"):
                available = self._pick_available(r, azure_models, provider_models, openrouter_models)
                if available:
                    print(f"empty route {decision.route}, falling back to {r}")
                    decision.route = r
                    break
        if not available:
            raise RuntimeError("No available models across all routes")

        model = self._resolve_with_llm(requested_model, decision, available)

        # Validate and prefix based on route
        if model not in available:
            # Last resort: fuzzy match when the model replies with a near miss or alias
            close = difflib.get_close_matches(model, available, n=1, cutoff=0.6)
            if close:
                print(f"resolver returned {model!r}, using closest available {close[0]!r}")
                model = close[0]
            else:
                raise RuntimeError(f"Resolved model {model!r} not in available options: {available!r}")
        return self._prefix(decision.route, model)

    # ---- Internals ----
    def _load_catalogs(self) -> tuple[List[str], List[str], List[str]]:
        """Load available models for each route with edge case handling."""
        # Azure models come from env var "AZURE_API_MODELS" as comma separated
        raw = os.getenv("AZURE_API_MODELS", "") or ""
        azure_models = [self._normalize(m) for m in raw.split(",") if m and m.strip()]

        # Direct provider models loaded from litellm catalog if relevant API keys are available
        try:
            provider_models_raw = litellm.utils.get_valid_models()
            provider_models = [self._normalize(m) for m in provider_models_raw if isinstance(m, str) and "openrouter" not in m.lower() and "azure" not in m.lower()]
        except Exception as e:  
            print(f"Failed to load provider models from litellm: {e}")
            provider_models = []

        # OpenRouter models list
        openrouter_models: List[str] = []
        try:
            if os.getenv("OPENROUTER_API_KEY"):
                client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY"))
                openrouter_models = [self._normalize(m.id) for m in client.models.list().data]
        except Exception as e:  # network
            print(f"Failed to fetch OpenRouter models: {e}")
            openrouter_models = []

        # Deduplicate and drop empties
        azure_models = sorted(set(filter(None, azure_models)))
        provider_models = sorted(set(filter(None, provider_models)))
        openrouter_models = sorted(set(filter(None, openrouter_models)))

        return azure_models, provider_models, openrouter_models

    def _decide_route(self, requested_model: str, azure_models: List[str], provider_models: List[str]) -> RouteDecision:
        """Ask a small model to select the route given facts.

        The judge must return strict JSON matching RouteDecision.
        """
        facts = {
            "requested_model": requested_model,
            "azure_models_available": azure_models,
            "provider_api_keys_available": [
                k for k, v in os.environ.items() if "API_KEY" in k and k not in {"OPENROUTER_API_KEY", "AZURE_API_KEY"} and v
            ],
            "provider_models_available": provider_models,
        }
        sys = (
            "You are a routing judge. Choose the single best route for a model request."
            "Rules:"
            "1. If the requested model semantically matches one of azure_models_available then you must prefer azure. Allow date style suffix matches (e.g. 'gpt-5-2025-11-16' matches 'gpt-5'), but do not allow cross version matches (e.g. 'gpt-4o' does not match 'gpt-4.1')."
            "2. Else if requested model matches a provider for which a direct API key is available then you must choose provider."
            "You can verify provider availability using provider_models_available."
            "3. Else choose openrouter. Default choice should be openrouter."
            "Return strict JSON with keys route and reason. Route must be one of azure, provider, openrouter. No extra text."
        )

        resp = self._with_retries(
            lambda: completion(
                model=self.routing_judge,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(facts, ensure_ascii=False)}],
                response_format=RouteDecision,
                timeout=self.request_timeout,
            )
        )

        content = (resp.choices[0].message.content or "").strip()
        try:
            return RouteDecision.model_validate_json(content)
        except Exception:
            return RouteDecision(route="openrouter", reason="Fallback due to invalid judge response")

    def _resolve_with_llm(self, requested_model: str, decision: RouteDecision, available_models: List[str]) -> str:
        """Ask a small model to map the requested model to one of the available models."""
        trimmed = self._prioritize_models(requested_model, available_models)
        sys = (
            "You are a model resolver."
            f"The requested model is {requested_model}."
            f"The requested model {requested_model} will be provided through {decision.route} because {decision.reason}."
            "You are given a list of available models."
            "You need to resolve the model to one of the available models."
            "Respond exactly with one of the available model options and nothing else."
        )
        resp = self._with_retries(
            lambda: completion(
                model=self.routing_judge,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(trimmed, ensure_ascii=False)}],
                timeout=self.request_timeout,
            )
        )

        content = (resp.choices[0].message.content or "").strip()
        if not content:
            raise RuntimeError("Empty resolver response")
        return self._normalize(content)

    # ---- Helpers ----
    @staticmethod
    def _normalize(name: str) -> str:
        # Normalize whitespace and case, but return lowercase ids for stable comparisons
        return (name or "").strip().replace(" ", "").lower()

    @staticmethod
    def _prefix(route: str, model: str) -> str:
        if route == "azure":
            return f"azure/{model}"
        if route == "openrouter":
            return f"openrouter/{model}"
        return model

    @staticmethod
    def _pick_available(route: str, azure: List[str], provider: List[str], openrouter: List[str]) -> List[str]:
        if route == "azure":
            return azure
        if route == "provider":
            return provider
        return openrouter

    @staticmethod
    def _find_exact(req: str, azure: List[str], provider: List[str], openrouter: List[str]):
        # Exact match by normalized model
        if req in azure:
            return ("azure", req)
        if req in provider:
            return ("provider", req)
        if req in openrouter:
            return ("openrouter", req)
        return None

    @staticmethod
    def _prioritize_models(requested: str, models: List[str]) -> List[str]:
        """Bring likely matches to the front and cap the list length. Put prefix matches first, then substring matches, then rest."""
        starts = [m for m in models if m.startswith(requested.split("-", 1)[0])]
        contains = [m for m in models if requested.split("-", 1)[0] in m and m not in starts]
        rest = [m for m in models if m not in starts and m not in contains]
        prioritized = starts + contains + rest
        return prioritized

    @staticmethod
    def _with_retries(fn, attempts: int = 3, base_sleep: float = 0.5):
        for i in range(attempts):
            try:
                return fn()
            except Exception as e:
                # Retry on transient errors like rate limits or timeouts
                if i == attempts - 1:
                    raise RuntimeError(f"LLM call failed after {attempts} attempts: {e}")
                sleep = base_sleep * (2 ** i)
                print(f"LLM call failed on attempt {i + 1}: {e}. Retrying in {sleep:.2f}s")
                time.sleep(sleep)

