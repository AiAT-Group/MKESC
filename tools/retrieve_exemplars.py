"""Build the hierarchical two-stage DPR exemplar annotations used by MKESC."""

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CandidateTurn:
    user_text: str
    persona_user_text: str
    response: str
    strategy: str

    def as_dpr_record(self):
        return [self.user_text, self.strategy, self.response]


def load_jsonl(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path, samples):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")


def is_user(turn):
    return isinstance(turn, dict) and turn.get("speaker") in {"usr", "user", "seeker"}


def is_system(turn):
    return isinstance(turn, dict) and turn.get("speaker") in {"sys", "system", "supporter"}


def turn_text(turn):
    value = turn.get("text", turn.get("content", ""))
    return value.strip() if isinstance(value, str) else ""


def strategy_text(turn):
    if isinstance(turn.get("strategy"), str):
        return turn["strategy"]
    annotation = turn.get("annotation", {})
    return annotation.get("strategy", "Others") if isinstance(annotation, dict) else "Others"


def persona_for_system_turn(sample, system_index):
    dialog = sample.get("dialog", [])
    persona_list = sample.get("persona_list", [])
    user_count = sum(1 for turn in dialog[: system_index + 1] if is_user(turn))
    if user_count <= 2 or user_count - 3 >= len(persona_list):
        return ""
    value = persona_list[user_count - 3]
    return value.strip() if isinstance(value, str) else ""


def extract_candidate_turns(sample):
    dialog = sample.get("dialog", [])
    turns = []
    for index, turn in enumerate(dialog):
        if index == 0 or not is_system(turn) or not is_user(dialog[index - 1]):
            continue
        user_text = turn_text(dialog[index - 1])
        response = turn_text(turn)
        if not user_text or not response:
            continue
        persona = persona_for_system_turn(sample, index)
        persona_user = " ".join(part for part in (persona, user_text) if part)
        turns.append(CandidateTurn(user_text, persona_user, response, strategy_text(turn)))
    return turns


class DPRScorer:
    def __init__(self, model_dir, device, batch_size, max_length):
        import torch
        from transformers import DPRReader, DPRReaderTokenizerFast

        self.torch = torch
        self.tokenizer = DPRReaderTokenizerFast.from_pretrained(model_dir)
        self.reader = DPRReader.from_pretrained(model_dir).to(device)
        self.reader.eval()
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

    def score(self, questions, passages):
        if len(questions) != len(passages):
            raise ValueError("questions and passages must have the same length")
        scores = []
        with self.torch.no_grad():
            for start in range(0, len(questions), self.batch_size):
                batch = self.tokenizer(
                    questions[start : start + self.batch_size],
                    passages[start : start + self.batch_size],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.device)
                scores.extend(self.reader(**batch).relevance_logits.detach().cpu().tolist())
        return scores


def top_indices(scores, k):
    return sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:k]


def first_stage_candidates(query_sample, training_samples, scorer, top_k):
    query = query_sample.get("situation", "")
    situations = [sample.get("situation", "") for sample in training_samples]
    if not query or not situations:
        return []
    scores = scorer.score([query] * len(situations), situations)
    return top_indices(scores, min(top_k, len(situations)))


def second_stage_exemplars(query_sample, candidate_samples, scorer, num_exemplars):
    candidate_turns = []
    for sample in candidate_samples:
        candidate_turns.extend(extract_candidate_turns(sample))
    if not candidate_turns:
        return []

    dialog = query_sample.get("dialog", [])
    exemplars_by_turn = {}
    for index, turn in enumerate(dialog):
        if index == 0 or not is_system(turn) or not is_user(dialog[index - 1]):
            continue
        user_text = turn_text(dialog[index - 1])
        if not user_text:
            continue
        current_response = turn_text(turn)
        persona = persona_for_system_turn(query_sample, index)
        query = " ".join(part for part in (persona, user_text) if part)
        eligible_candidates = [
            candidate
            for candidate in candidate_turns
            if not (candidate.user_text == user_text and candidate.response == current_response)
        ]
        if not eligible_candidates:
            eligible_candidates = candidate_turns
        scores = scorer.score(
            [query] * len(eligible_candidates),
            [candidate.persona_user_text for candidate in eligible_candidates],
        )
        selected = top_indices(scores, min(num_exemplars, len(eligible_candidates)))
        exemplars_by_turn[index] = [eligible_candidates[i].as_dpr_record() for i in selected]
    return exemplars_by_turn


def build_retrieval_annotations(query_samples, training_samples, scorer, first_stage_k, num_exemplars):
    annotated = []
    for sample in tqdm(query_samples, desc="Building DPR exemplars"):
        result = copy.deepcopy(sample)
        candidate_indices = first_stage_candidates(result, training_samples, scorer, first_stage_k)
        candidate_samples = [training_samples[index] for index in candidate_indices]
        exemplars = second_stage_exemplars(result, candidate_samples, scorer, num_exemplars)
        for system_index, records in exemplars.items():
            result["dialog"][system_index]["dpr"] = records
        annotated.append(result)
    return annotated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-file", required=True, help="JSONL split to annotate")
    parser.add_argument("--retrieval-corpus", required=True, help="Training JSONL split used as the retrieval corpus")
    parser.add_argument("--output", required=True, help="Output JSONL file with dpr fields")
    parser.add_argument("--dpr-model-dir", default="./DPR-reader")
    parser.add_argument("--device", default=None, help="PyTorch device; defaults to CUDA when available")
    parser.add_argument("--first-stage-k", type=int, default=10)
    parser.add_argument("--num-exemplars", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=32)
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    query_samples = load_jsonl(args.query_file)
    training_samples = load_jsonl(args.retrieval_corpus)
    scorer = DPRScorer(args.dpr_model_dir, device, args.batch_size, args.max_length)
    annotated = build_retrieval_annotations(
        query_samples,
        training_samples,
        scorer,
        args.first_stage_k,
        args.num_exemplars,
    )
    write_jsonl(args.output, annotated)


if __name__ == "__main__":
    main()
