import argparse
import json

def compute_meteor(input_file):
    from nltk.translate.meteor_score import meteor_score

    scores = []

    with open(input_file, encoding="utf-8") as file:
        for line in file:
            sample = json.loads(line)
            reference = sample["response"].split()
            hypothesis = sample["generation"].split()
            scores.append(meteor_score([reference], hypothesis))

    if not scores:
        raise ValueError("The input file does not contain any samples.")
    return sum(scores) / len(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute METEOR for a JSONL result file.")
    parser.add_argument("input_file", help="JSONL file containing response and generation fields")
    args = parser.parse_args()
    print(f"METEOR: {compute_meteor(args.input_file) * 100:.2f}")
