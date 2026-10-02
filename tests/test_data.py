from llmornot.data import make_splits, normalise


def test_normalise_ignores_case_punctuation_and_spacing():
    assert normalise("  Why was my card   DECLINED?! ") == "why was my card declined"
    assert normalise("\nI can't top-up.") == "i cant topup"


def test_splits_are_disjoint_stratified_and_reproducible():
    s1, s2 = make_splits(), make_splits()
    # train and val are carved from the same file: no row may appear in both
    assert set(s1.train.row_id).isdisjoint(s1.val.row_id)
    assert len(s1.train) + len(s1.val) == 10003
    # every intent must appear in every split, otherwise per-class metrics break
    for df in (s1.train, s1.val, s1.test, s1.llm_sample):
        assert df.label.nunique() == 77
    assert len(s1.llm_sample) == 500 and set(s1.llm_sample.row_id) <= set(s1.test.row_id)
    # same seed -> identical splits
    assert s1.val.row_id.tolist() == s2.val.row_id.tolist()
    assert s1.llm_sample.row_id.tolist() == s2.llm_sample.row_id.tolist()
