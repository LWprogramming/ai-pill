"""
Common utilities for handling response data across collection and analysis scripts.
"""

import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
from enum import Enum

PROMPTS_FILE = "prompts.json"
RESPONSES_DIR = "responses"


class AnswerEnum(Enum):
    BLUE = "blue"
    RED = "red"
    REFUSAL = "refusal"
    OTHER = "other"


def is_valid_answer_enum(answer: Any) -> bool:
    try:
        AnswerEnum(answer)  # either string or enum itself.
        return True
    except ValueError:
        return False


def load_prompts(prompts_file: str = PROMPTS_FILE) -> Dict:
    """Load and parse prompts from the prompts file"""
    with open(prompts_file, "r") as file:
        prompts = json.load(file)["scenarios"]
        # Create a flat dictionary of prompt_id -> prompt
        return {
            variation["id"]: {"scenario": scenario, "variation": variation}
            for scenario in prompts
            for variation in scenario["variations"]
        }


def get_response_path(llm: str, prompt_id: str) -> str:
    """Create and return path to responses for a specific LLM and prompt"""
    path = os.path.join(RESPONSES_DIR, llm, prompt_id)
    os.makedirs(path, exist_ok=True)
    return path


def save_response(response_obj: Dict[str, Any]) -> None:
    """Save a single response to a unique file"""
    llm_name = response_obj["llm"]
    prompt_id = response_obj["prompt_id"]

    # Create unique filename with timestamp and UUID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"response_{timestamp}_{unique_id}.json"

    response_dir = get_response_path(llm_name, prompt_id)
    response_path = os.path.join(response_dir, filename)

    with open(response_path, "w") as f:
        json.dump(response_obj, f, indent=2)


def load_responses(
    llm: Optional[str] = None, prompt_id: Optional[str] = None
) -> List[Dict]:
    """
    Load all response files from the responses directory.
    If llm and/or prompt_id provided, only load matching responses.
    """
    responses = []
    base_path = RESPONSES_DIR
    os.makedirs(base_path, exist_ok=True)

    # If both llm and prompt_id specified, only check that specific directory
    if llm and prompt_id:
        response_dir = os.path.join(base_path, llm, prompt_id)
        if os.path.exists(response_dir):
            for f in os.listdir(response_dir):
                if f.startswith("response_") and f.endswith(".json"):
                    try:
                        with open(os.path.join(response_dir, f)) as file:
                            responses.append(json.load(file))
                    except Exception as e:
                        print(f"Error loading {f}: {e}")
        return responses

    # Otherwise load all responses
    for llm_dir in os.listdir(base_path):
        if not os.path.isdir(os.path.join(base_path, llm_dir)):
            continue
        if llm and llm_dir != llm:
            continue

        for pid in os.listdir(os.path.join(base_path, llm_dir)):
            if prompt_id and pid != prompt_id:
                continue

            response_dir = os.path.join(base_path, llm_dir, pid)
            if not os.path.isdir(response_dir):
                continue

            for f in os.listdir(response_dir):
                if f.startswith("response_") and f.endswith(".json"):
                    try:
                        with open(os.path.join(response_dir, f)) as file:
                            responses.append(json.load(file))
                    except Exception as e:
                        print(f"Error loading {f}: {e}")

    return responses


def count_existing_responses(llm: str, prompt_id: str) -> int:
    """Count number of existing responses for a given LLM and prompt combination"""
    response_dir = get_response_path(llm, prompt_id)
    return len(
        [
            f
            for f in os.listdir(response_dir)
            if f.startswith("response_") and f.endswith(".json")
        ]
    )
