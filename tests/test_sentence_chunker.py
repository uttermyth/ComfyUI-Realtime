# tests/test_sentence_chunker.py
from comfyui_realtime.engine.sentence_chunker import SentenceChunker


def test_feed_returns_nothing_until_a_boundary():
    chunker = SentenceChunker()
    assert chunker.feed("Hello") == []
    assert chunker.feed(" there") == []


def test_feed_returns_a_complete_sentence_once_boundary_arrives():
    chunker = SentenceChunker()
    chunker.feed("Hello there")
    sentences = chunker.feed(".")
    assert sentences == ["Hello there."]


def test_feed_handles_multiple_sentences_in_one_call():
    chunker = SentenceChunker()
    sentences = chunker.feed("Hi! How are you? I am fine.")
    assert sentences == ["Hi!", "How are you?", "I am fine."]


def test_feed_accumulates_across_many_small_chunks():
    chunker = SentenceChunker()
    text = "This is one sentence. This is another!"
    collected = []
    for ch in text:
        collected.extend(chunker.feed(ch))
    assert collected == ["This is one sentence.", "This is another!"]


def test_flush_returns_trailing_partial_text():
    chunker = SentenceChunker()
    chunker.feed("No boundary yet")
    assert chunker.flush() == "No boundary yet"


def test_flush_returns_none_when_nothing_pending():
    chunker = SentenceChunker()
    chunker.feed("Complete.")
    assert chunker.flush() is None
