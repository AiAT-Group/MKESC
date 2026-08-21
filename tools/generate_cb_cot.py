"""Generate offline cognitive-affective annotations for the MKESC pipeline."""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path


LOGGER = logging.getLogger(__name__)
FIELDS = (
    "persona_situation",
    "persona_thought",
    "persona_emotion",
    "persona_behavior",
)

PROMPT_TEMPLATE = """
[Role Definition]
You are an emotional support expert who infers a help-seeker's internal state.

[Task Description]
Infer the help-seeker's situation, thought, emotion, and behavior from the current
user utterance. Use the dialogue history only for contextual awareness and coherence
checking. Do not use a previous user utterance as the primary basis for inference.

[Required Reasoning]
1. Situation: summarize the problem or challenge.
2. Thought: infer the help-seeker's appraisal, belief, expectation, or concern.
3. Emotion: output 1-3 concise emotion labels, not a sentence.
4. Behavior: infer observable or implied behavior, coping patterns, or action intent.

All inferences must be grounded in the current user utterance. Do not introduce facts
that are not supported by the dialogue.

[Output Format]
Return a JSON object with exactly these fields:
{{
  "persona_situation": "one concise sentence",
  "persona_thought": "one concise sentence",
  "persona_emotion": ["emotion label"],
  "persona_behavior": "one concise sentence"
}}

[Dialogue History]
{dialogue_history}

[Current User Utterance]
user: {current_utterance}
""".strip()


def normalize_role(speaker):
    if speaker == "usr":
        return "user"
    if speaker == "sys":
        return "system"
    return str(speaker)


def find_previous_user(dialog, system_index):
    for index in range(system_index - 1, -1, -1):
        turn = dialog[index]
        if not isinstance(turn, dict) or turn.get("speaker") != "usr":
            continue
        text = turn.get("text", "")
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None


def build_dialogue_history(dialog, system_index):
    lines = []
    for turn in dialog[:system_index]:
        if not isinstance(turn, dict):
            continue
        text = turn.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        lines.append(f"{normalize_role(turn.get('speaker'))}: {text.strip()}")
    return "\n".join(lines) if lines else "{}"


def already_annotated(turn):
    return all(field in turn for field in FIELDS)


def write_fallback(turn):
    turn["persona_situation"] = "unknown"
    turn["persona_thought"] = "unknown"
    turn["persona_emotion"] = ["unknown"]
    turn["persona_behavior"] = "unknown"


def parse_json_response(content):
    """Parse a JSON object from a plain-text or fenced model response."""
    if not isinstance(content, str):
        raise ValueError("The LLM response did not contain text content")

    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("The LLM response did not contain a JSON object")
    return json.loads(cleaned[start:end + 1])


def call_llm(client, model, user_text, history, retries, temperature):
    prompt = PROMPT_TEMPLATE.format(
        dialogue_history=history,
        current_utterance=user_text,
    )
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=512,
            )
            parsed = parse_json_response(response.choices[0].message.content)
            if not all(field in parsed for field in FIELDS):
                raise ValueError(f"Missing one or more required fields: {FIELDS}")
            return parsed
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(min(20, 2**attempt + random.random()))


def process_file(input_path, output_path, model, base_url, api_key, retries, temperature):
    from openai import OpenAI
    from tqdm import tqdm

    if not api_key:
        raise ValueError("Provide --api-key or set SILICONFLOW_API_KEY.")

    client = OpenAI(base_url=base_url, api_key=api_key)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding="utf-8") as source:
        total_lines = sum(1 for line in source if line.strip())

    with open(input_path, encoding="utf-8") as source, open(output_path, "w", encoding="utf-8") as target:
        for line in tqdm(source, total=total_lines, desc="Generating CB-CoT annotations"):
            if not line.strip():
                continue
            sample = json.loads(line)
            dialog = sample.get("dialog", [])
            if not isinstance(dialog, list):
                target.write(json.dumps(sample, ensure_ascii=False) + "\n")
                continue

            for system_index, turn in enumerate(dialog):
                if not isinstance(turn, dict) or turn.get("speaker") != "sys":
                    continue
                if already_annotated(turn):
                    continue

                user_text = find_previous_user(dialog, system_index)
                if user_text is None:
                    continue

                history = build_dialogue_history(dialog, system_index)
                try:
                    turn.update(call_llm(client, model, user_text, history, retries, temperature))
                except Exception as error:
                    LOGGER.warning("CB-CoT failed at system turn %s: %s", system_index, error)
                    write_fallback(turn)

            target.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL file in the MKESC dialogue format")
    parser.add_argument("--output", required=True, help="Output JSONL file with CB-CoT fields")
    parser.add_argument(
        "--model",
        default=os.environ.get("MKESC_CB_COT_MODEL"),
        help="Exact OpenAI-compatible model ID confirmed for the paper's CB-CoT stage",
    )
    parser.add_argument("--base-url", default="https://api.siliconflow.cn/v1")
    parser.add_argument("--api-key", default=os.environ.get("SILICONFLOW_API_KEY"))
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    if not args.model:
        parser.error("Provide --model or set MKESC_CB_COT_MODEL.")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    process_file(
        args.input,
        args.output,
        args.model,
        args.base_url,
        args.api_key,
        args.retries,
        args.temperature,
    )


if __name__ == "__main__":
    main()
