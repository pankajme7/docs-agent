import json
from pathlib import Path


def test_evaluation_dataset_has_twenty_questions():
    path = Path(__file__).resolve().parents[1] / "data/evaluations/questions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 20
    assert len({row["id"] for row in rows}) == 20
    assert any(not row["answerable"] for row in rows)
