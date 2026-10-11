from engine.chat import StopSequenceFilter


def feed_all(stop: list[str], pieces: list[str]) -> tuple[str, str | None]:
    f = StopSequenceFilter(stop)
    out = "".join(f.feed(p) for p in pieces) + f.flush()
    return out, f.matched


def test_passes_text_without_stop_sequences():
    assert feed_all([], ["Hello", " world"]) == ("Hello world", None)


def test_cuts_at_stop_sequence_split_across_pieces():
    assert feed_all(["END"], ["one two E", "ND three"]) == ("one two ", "END")


def test_releases_held_back_partial_match():
    assert feed_all(["END"], ["say E", "N"]) == ("say EN", None)
