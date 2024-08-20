"""Test function to test the count of tokens."""

from llmax import tokens


def test_token_count_short() -> None:
    """For several phrases, test whether the number of tokens is correct."""
    data = [
        ("J'aime les légumes.", 8),
        ("Another sentence.", 4),
        (
            "To iterate over all the items in your data list and perform an assertion for each, you can use a loop within your test. This allows you to check that the count of tokens for each string in data matches the expected value.",
            47,
        ),
    ]

    for sentence, expected_count in data:
        assert tokens.count(sentence) == expected_count
