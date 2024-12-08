"""
Script to collect responses from various LLMs for a set of prompts.
"""

import os
import time
import traceback
from typing import Any, Dict, Optional, List
from concurrent.futures import ProcessPoolExecutor
import dirtyjson
import google.generativeai as genai
from openai import OpenAI
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
import anthropic
import typing_extensions
from datetime import datetime
from google.api_core.exceptions import ResourceExhausted

from response_utils import (
    PROMPTS_FILE,
    AnswerEnum,
    is_valid_answer_enum,
    load_prompts,
    save_response,
    load_responses,
)

# DEBUG = True
DEBUG = False
GEMINI_API_KEY = os.getenv("PERSONAL_GOOGLE_AISTUDIO_API_KEY")
assert GEMINI_API_KEY, "GEMINI_API_KEY is not set"
ANTHROPIC_API_KEY = os.getenv("PERSONAL_ANTHROPIC_API_KEY")
assert ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY is not set"
OPENAI_API_KEY = os.getenv("PERSONAL_OPENAI_KEY")
assert OPENAI_API_KEY, "OPENAI_API_KEY is not set"

genai.configure(api_key=GEMINI_API_KEY)


# Configurable variables
class LLMConfig(typing_extensions.TypedDict):
    name: str
    rate_limit_delay: float  # Delay in seconds between requests
    model_provider: str  # "openai" | "anthropic" | "google"

# really restrictive rate limits because we're using flash for ALL the cleaning so we're just gonna get rate limited on everything if our gemini thing stops working. but who cares about the absolute speed? as long as it works.
LLMS: List[LLMConfig] = [
    # openai line
    # {"name": "gpt-4o-2024-11-20", "rate_limit_delay": 30.0, "model_provider": "openai"},
    # {"name": "gpt-4o-2024-08-06", "rate_limit_delay": 30.0, "model_provider": "openai"},
    {
        "name": "gpt-4o-mini-2024-07-18",
        "rate_limit_delay": 30.0,
        "model_provider": "openai",
    },
    # {"name": "gpt-4-0613", "rate_limit_delay": 30.0, "model_provider": "openai"},
    # {"name": "gpt-4-0314", "rate_limit_delay": 30.0, "model_provider": "openai"},
    # {
    #     "name": "o1-preview-2024-09-12",
    #     "rate_limit_delay": 30.0,
    #     "model_provider": "openai",
    # },
    # {"name": "o1-mini-2024-09-12", "rate_limit_delay": 30.0, "model_provider": "openai"},
    # todo: we want to get all the models but let's pick a cheap one just to dip toes in
    # claude line
    # {
    #     "name": "claude-3-opus-20240229",
    #     "rate_limit_delay": 30.0,
    #     "model_provider": "anthropic",
    # },
    # {
    #     "name": "claude-3-5-sonnet-20241022",
    #     "rate_limit_delay": 30.0,
    #     "model_provider": "anthropic",
    # },
    # {
    #     "name": "claude-3-5-haiku-20241022",
    #     "rate_limit_delay": 30.0,
    #     "model_provider": "anthropic",
    # },
    # gemini line
    # todo: seems like we don't have this yet? idk
    # {"name": "gemini-exp-1121", "rate_limit_delay": 30.0, "model_provider": "google"},
    # {
    #     "name": "gemini-1.5-flash-002",
    #     "rate_limit_delay": 30.0,
    #     "model_provider": "google",
    # },
]

NUM_RESPONSES_PER_LLM = 30

# region response cleaning
RESPONSE_CLEANING_SYSTEM_PROMPT = """
You are a helpful assistant that extracts the final choice and reasoning from an LLM's response text.

The response will be about choosing between pills (either red vs blue).

Extract:
1. The final stated choice 
2. The explanation/reasoning given

Format the output as JSON with this structure:
{
    "answer": "blue" | "red" | "refusal" | "other",
    "explanation": "1-2 sentence summary of why the choice was made, based on input text."
}

If no clear choice is stated or the response refuses to choose, use "refusal".
If the choice doesn't fit the above categories, use "other".
"""


class GeminiCleanedResponse(typing_extensions.TypedDict):
    answer: AnswerEnum
    explanation: str


class FullCleanedResponse(typing_extensions.TypedDict):
    gemini_cleaned_response: GeminiCleanedResponse
    raw_response: str  # The raw response text


def is_gemini_cleaned_response(response: Dict[str, Any]) -> bool:
    print(f"Checking GeminiCleanedResponse of type {type(response)}: {response}")
    conditions_and_error_messages = [
        (is_valid_answer_enum(response["answer"]), "answer is not an AnswerEnum"),
        (isinstance(response["explanation"], str), "explanation is not a string"),
    ]
    errors = [msg for cond, msg in conditions_and_error_messages if not cond]
    if errors:
        print(f"Invalid GeminiCleanedResponse: {', '.join(errors)}")
        return False
    return True


def clean_llm_response(raw_response: str) -> Optional[FullCleanedResponse]:
    """
    Clean a raw response from an LLM using Gemini's cleaning API.
    Returns a FullCleanedResponse or None if an error occurs.
    """
    print(f"Cleaning response of type {type(raw_response)}: {raw_response}")
    if DEBUG:
        return {
            "gemini_cleaned_response": {
                "answer": AnswerEnum.OTHER.value,
                "explanation": "dummy explanation"
            },
            "raw_response": raw_response
        }
    try:
        model = genai.GenerativeModel(
            "gemini-1.5-flash-8b",
            system_instruction=RESPONSE_CLEANING_SYSTEM_PROMPT,
        )
        response = model.generate_content(
            raw_response,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            },
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=GeminiCleanedResponse,
                temperature=0.0,
            ),
        )
        response_text: str = response.text
        cleaned_response = dirtyjson.loads(response_text)
        if not is_gemini_cleaned_response(cleaned_response):
            raise ValueError("Cleaned response is not a GeminiCleanedResponse")
        return {
            "gemini_cleaned_response": cleaned_response,
            "raw_response": raw_response,
        }
    except ResourceExhausted as e:
        print(f"Rate limit exceeded: {e}")
        time.sleep(60.0) # probably just 429 too many requests, chill out for like a minute to let the quota refresh.
        return None
    except Exception as e:
        print(f"Failed to clean response: {traceback.format_exc()}")
        return None
# endregion response cleaning


# region LLM API calls
class LLMResponse(typing_extensions.TypedDict):
    content: str  # The actual response text
    model: str  # The specific model version used
    provider: str  # The provider (openai, anthropic, gemini)
    metadata: Dict[str, Any]  # Usage stats, model info, etc (optional)


def call_openai(prompt_text: str, model_version: str = "gpt-4o") -> Optional[LLMResponse]:
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model=model_version,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.0,
        )
        return {
            "content": response.choices[0].message.content,
            "model": model_version,
            "provider": "openai",
            "metadata": {
                "usage": {
                    "total_tokens": response.usage.total_tokens,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                }
            }
        }
    except Exception as e:
        print(f"Error calling OpenAI API: {traceback.format_exc()}")
        return None


def call_anthropic(prompt_text: str, model_version: str = "claude-3-5-sonnet-20241022") -> Optional[LLMResponse]:
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=model_version,
            max_tokens=4096,
            temperature=0,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
        )
        return {
            "content": response.content[0].text,
            "model": model_version,
            "provider": "anthropic",
            "metadata": {}
        }
    except Exception as e:
        print(f"Error calling Anthropic: {traceback.format_exc()}")
        return None


def call_gemini(prompt_text: str, gemini_model_version: str = "gemini-exp-1121") -> Optional[LLMResponse]:
    try:
        model = genai.GenerativeModel(gemini_model_version)
        response = model.generate_content(
            prompt_text,
            safety_settings={
                # apparently OFF is not a valid setting even though the enum is there in the code because there's ANOTHER dict doing the same thing idk
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            },
            generation_config=genai.types.GenerationConfig(temperature=0.0),
        )
        return {
            "content": response.text,
            "model": gemini_model_version,
            "provider": "google",
            "metadata": {}
        }
    except Exception as e:
        print(f"Error calling Gemini: {traceback.format_exc()}")
        return None


def placeholder_call_llm_api(llm_config: LLMConfig, prompt_text: str) -> Optional[LLMResponse]:
    return {
        "content": "dummy response",
        "model": llm_config["name"],
        "provider": llm_config["model_provider"],
        "metadata": {}
    }


def call_llm_api(llm_config: LLMConfig, prompt_text: str) -> Optional[LLMResponse]:
    try:
        llm_config_name = llm_config["name"]
        llm_config_model_provider = llm_config["model_provider"]
        if DEBUG:
            return placeholder_call_llm_api(llm_config, prompt_text)
        elif llm_config_model_provider == "openai":
            return call_openai(prompt_text, model_version=llm_config_name)
        elif llm_config_model_provider == "anthropic":
            return call_anthropic(prompt_text, model_version=llm_config_name)
        elif llm_config_model_provider == "google":
            return call_gemini(prompt_text, gemini_model_version=llm_config_name)
        else:
            raise ValueError(f"LLM '{llm_config['name']}' is not supported.")
    except Exception as e:
        print(f"Error calling {llm_config['name']}: {traceback.format_exc()}")
        return None


def collect_responses_by_provider(provider: str, llm_configs: List[LLMConfig], prompts, existing_responses):
    """Collect responses for all models from a single provider, respecting rate limits"""
    provider_configs = [cfg for cfg in llm_configs if cfg["model_provider"] == provider]
    
    for llm_config in provider_configs:
        for prompt_id, prompt_data in prompts.items():
            variation = prompt_data["variation"]
            existing_count = len([
                r for r in existing_responses
                if r["llm"] == llm_config["name"] and r["prompt_id"] == prompt_id
            ])
            
            for i in range(existing_count, NUM_RESPONSES_PER_LLM):
                print(f"Collecting response {i+1} for LLM '{llm_config['name']}' on prompt '{prompt_id}'")
                
                raw_response = call_llm_api(llm_config, variation["prompt"])
                if raw_response:
                    cleaned_response = clean_llm_response(raw_response["content"])
                    if cleaned_response:
                        response_obj = {
                            "llm": llm_config["name"],
                            "prompt_id": prompt_id,
                            "response_number": i + 1,
                            "response": cleaned_response,
                            "metadata": raw_response["metadata"],
                            "timestamp": datetime.now().isoformat(),
                        }
                        save_response(response_obj)
                
                # Use provider-specific rate limit
                if not DEBUG:
                    time.sleep(llm_config["rate_limit_delay"])

def main():
    try:
        prompts = load_prompts(PROMPTS_FILE)
        responses = load_responses()
        
        # Create one worker per provider
        providers = ["openai", "anthropic", "google"]
        with ProcessPoolExecutor(max_workers=len(providers)) as executor:
            futures = [
                executor.submit(collect_responses_by_provider, provider, LLMS, prompts, responses)
                for provider in providers
            ]
            for future in futures:
                future.result()
    except Exception as e:
        print(f"Error in main: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
