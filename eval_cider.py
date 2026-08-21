import argparse
import json

def compute_cider(input_file):
    import nltk
    from metric.pycocoevalcap.cider.cider import Cider

    references = {}
    hypotheses = {}

    with open(input_file, encoding="utf-8") as file:
        for sample_id, line in enumerate(file):
            sample = json.loads(line)
            references[sample_id] = [
                " ".join(nltk.word_tokenize(sample["response"].lower()))
            ]
            hypotheses[sample_id] = [
                " ".join(nltk.word_tokenize(sample["generation"].lower()))
            ]

    score, _ = Cider().compute_score(references, hypotheses)
    return float(score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CIDEr for a JSONL result file.")
    parser.add_argument("input_file", help="JSONL file containing response and generation fields")
    args = parser.parse_args()
    print(f"CIDEr: {compute_cider(args.input_file) * 100:.2f}")
