import pytest

from llmornot.labels import INVALID, build_system_prompt, parse_label

LABELS = ["card_arrival", "Refund_not_showing_up", "reverted_card_payment?", "top_up_failed"]


@pytest.mark.parametrize("raw, expected", [
    ('{"intent": "card_arrival"}', (0, "exact")),
    ("card_arrival", (0, "exact")),
    ('{"intent": "reverted_card_payment?"}', (2, "exact")),      # label with a literal "?"
    ('{"intent": "Refund_not_showing_up"}', (1, "exact")),       # label with a capital letter
    ('{"intent": "refund_not_showing_up"}', (1, "normalised")),  # wrong case
    ("Top up failed.", (3, "normalised")),                       # spaces + trailing full stop
    ('"reverted_card_payment"', (2, "normalised")),             # JSON string, "?" dropped
    ("top-up_failed", (3, "normalised")),
])
def test_parse_valid_answers(raw, expected):
    assert parse_label(raw, LABELS) == expected


@pytest.mark.parametrize("raw", [
    "", "card", '{"intent": "card_lost"}', '{"label": "card_arrival"}', "I think it is card_arrival",
    '{"intent": 3}', "{not json",
])
def test_parse_invalid_answers(raw):
    assert parse_label(raw, LABELS) == (INVALID, "invalid")


def test_prompt_lists_every_label_alphabetically():
    names = build_system_prompt("zeroshot_names", LABELS)
    assert names.index("- card_arrival") < names.index("- Refund_not_showing_up") < names.index("- top_up_failed")
    described = build_system_prompt("zeroshot_descriptions", LABELS)
    assert "- card_arrival: " in described and "{label_block}" not in described
