import json

import pytest

from shadow_judge import CorpusError, Turn, dump, from_dicts, load


def write(tmp_path, payload):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_loads_the_envelope(tmp_path):
    path = write(tmp_path, {"version": 1, "turns": [{"message": "hi", "answers": ["yo"]}]})
    turns = load(path)
    assert len(turns) == 1
    assert turns[0].message == "hi"
    assert turns[0].answered


def test_loads_a_bare_list_so_older_exports_still_work(tmp_path):
    path = write(tmp_path, [{"message": "hi", "answers": ["yo"]}])
    assert len(load(path)) == 1


def test_a_single_answer_string_is_accepted():
    assert from_dicts([{"message": "hi", "answers": "just one"}])[0].answers == ["just one"]


def test_unknown_keys_are_carried_not_dropped():
    turn = from_dicts([{"message": "hi", "answers": [], "faq_id": 7, "channel": "whatsapp"}])[0]
    assert turn.context == {"faq_id": 7, "channel": "whatsapp"}


def test_escalated_defaults_to_false():
    assert from_dicts([{"message": "hi", "answers": ["a"]}])[0].escalated is False


def test_a_turn_with_no_answers_is_not_answered():
    assert not from_dicts([{"message": "hi", "answers": []}])[0].answered


@pytest.mark.parametrize(
    "row, complaint",
    [
        ({"answers": ["a"]}, "no usable 'message'"),
        ({"message": "   ", "answers": ["a"]}, "no usable 'message'"),
        ({"message": "hi", "answers": [1, 2]}, "not a list of strings"),
        ("just a string", "expected an object"),
    ],
)
def test_bad_rows_say_which_turn_and_why(row, complaint):
    with pytest.raises(CorpusError, match=complaint):
        from_dicts([row])


def test_missing_file_is_a_corpus_error(tmp_path):
    with pytest.raises(CorpusError, match="no corpus at"):
        load(tmp_path / "nope.json")


def test_invalid_json_says_so(tmp_path):
    path = tmp_path / "corpus.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CorpusError, match="not valid JSON"):
        load(path)


def test_an_empty_corpus_is_refused(tmp_path):
    with pytest.raises(CorpusError, match="no turns"):
        load(write(tmp_path, {"version": 1, "turns": []}))


def test_wrong_shape_is_refused(tmp_path):
    with pytest.raises(CorpusError, match="must hold a list of turns"):
        load(write(tmp_path, {"version": 1}))


def test_roundtrip(tmp_path):
    original = [
        Turn(id="a", message="hola", answers=["sí"], escalated=True, context={"channel": "ig"}),
        Turn(message="unanswered", answers=[]),
    ]
    path = tmp_path / "out.json"
    dump(original, path)
    reloaded = load(path)
    assert [t.message for t in reloaded] == ["hola", "unanswered"]
    assert reloaded[0].escalated is True
    assert reloaded[0].context == {"channel": "ig"}
    assert reloaded[1].answers == []
