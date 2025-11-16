import litellm
from litellm import completion
from dotenv import load_dotenv
from answers import Answer, Intervention
load_dotenv()

answer_class = Answer[Intervention]
answer_schema = answer_class.model_json_schema()  # Pydantic v2

tool_schema = {
    "type": "function",
    "function": {
        "name": "answer",
        "description": "Respond to the question by providing a decision and an explanation.",
        "parameters": answer_schema,
    },
}

# print(litellm.utils.get_valid_models())

for model in [
    "azure/gpt-5",
    "azure/gpt-5-mini",
    "azure/gpt-4.1",
    "azure/gpt-4.1-mini",
    "azure/gpt-4.1-nano",
    "openrouter/openai/gpt-3.5-turbo",
]: 
    print(f"Testing {model}...")
    response = completion(
        model=model,
        messages=[{"role": "user", "content": "Tell me a joke about cats"}]
    )
    print(response['choices'][0]['message']['content'])
    response = completion(
        model=model,
        messages=[{"role": "user", "content": "Should I get a flu shot? You're a doctor. Respond Intervene or Monitor."}], 
        tools=[tool_schema],
        tool_choice={"type": "function", "function": {"name": "answer"}},
    )
    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        args = tool_calls[0].function.arguments
        answer = answer_class.model_validate_json(args)
        print(answer.decision)
    if response.get("model") != model.removeprefix("openrouter/"):
        print(f"❌ failed! Model actually used: {response.get('model')}")
    else: print(f"✅ works!")
    print()