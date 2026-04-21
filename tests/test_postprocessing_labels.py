from neuropyguin.tabs.postprocessing_tab import _is_bombcell_good_label


def test_bombcell_good_filter_accepts_somatic_and_non_somatic_good_labels() -> None:
    assert _is_bombcell_good_label("good")
    assert _is_bombcell_good_label("GOOD")
    assert _is_bombcell_good_label("non_soma")
    assert _is_bombcell_good_label("NON-SOMA")
    assert _is_bombcell_good_label("NON-SOMA GOOD")
    assert _is_bombcell_good_label("non soma")


def test_bombcell_good_filter_rejects_non_good_labels() -> None:
    assert not _is_bombcell_good_label("mua")
    assert not _is_bombcell_good_label("noise")
    assert not _is_bombcell_good_label("")
    assert not _is_bombcell_good_label(None)
