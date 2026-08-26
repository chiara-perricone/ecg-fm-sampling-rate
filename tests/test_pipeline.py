"""Il collegamento fra ``configs/runs.csv`` e i dati veri.

Due sono le cose che qui possono rompersi in silenzio: le colonne del set
congelato e la scelta dello scaler. Entrambe darebbero numeri plausibili se
sbagliate, il che e' esattamente il motivo per cui hanno dei test.
"""

import json

import pytest

torch = pytest.importorskip("torch")  # pipeline importa ecgres.model
pytest.importorskip("pandas")
pytest.importorskip("scipy")

from ecgres.pipeline import clean_columns, git_sha, scaler_spec  # noqa: E402
from ecgres.runs import enumerate_runs  # noqa: E402

ALL_LABELS = [f"L{i:02d}" for i in range(71)]
CLEAN_LABELS = ALL_LABELS[:49]


def _frozen(tmp_path, **overrides):
    payload = {
        "min_test_positives": 10,
        "test_fold": 10,
        "n_labels_all": 71,
        "n_labels_clean": 49,
        "labels_clean": CLEAN_LABELS,
        **overrides,
    }
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_clean_columns_maps_names_to_positions(tmp_path):
    columns = clean_columns(ALL_LABELS, _frozen(tmp_path))
    assert len(columns) == 49
    assert [ALL_LABELS[i] for i in columns] == CLEAN_LABELS


def test_clean_columns_preserves_the_frozen_order(tmp_path):
    """L'ordine e' quello del file, non quello della matrice: entrambe le
    metriche devono indicizzare le stesse colonne nello stesso ordine."""
    shuffled = list(reversed(CLEAN_LABELS))
    path = _frozen(tmp_path, labels_clean=shuffled)
    columns = clean_columns(ALL_LABELS, path)
    assert [ALL_LABELS[i] for i in columns] == shuffled


def test_clean_columns_rejects_a_self_inconsistent_file(tmp_path):
    path = _frozen(tmp_path, n_labels_clean=50)
    with pytest.raises(ValueError, match="dichiara 50"):
        clean_columns(ALL_LABELS, path)


def test_clean_columns_rejects_a_matrix_of_another_width(tmp_path):
    """Se scp_statements.csv cambiasse, il set congelato non e' piu' applicabile."""
    with pytest.raises(ValueError, match="colonne"):
        clean_columns(ALL_LABELS[:70], _frozen(tmp_path))


def test_clean_columns_rejects_an_absent_label(tmp_path):
    labels = list(ALL_LABELS)
    labels[0] = "ALTRO"
    with pytest.raises(ValueError, match="assenti"):
        clean_columns(labels, _frozen(tmp_path))


# --------------------------------------------------------------------------
# Scaler (§10 voci 7 e 8)
# --------------------------------------------------------------------------


def _by_block(block):
    return [r for r in enumerate_runs() if r.block == block]


def test_stage0_fits_on_its_own_cache():
    """Voce 8: *Stage 0 uses the reference pipeline's own scaler*."""
    for run in _by_block(0):
        assert scaler_spec(run) == (100, False) == (run.fs, run.notch)


@pytest.mark.parametrize("block", [1, 2, 3, 4])
def test_the_comparison_shares_one_scaler_at_500hz_notched(block):
    """Un solo scaler per tutti i bracci, fittato dove nascono (voce 8)."""
    specs = {scaler_spec(run) for run in _by_block(block)}
    assert specs == {(500, True)}


def test_block_four_shares_the_scaler_with_block_three():
    """§8.3.2 deve isolare la provenienza, non la normalizzazione.

    I blocchi 3 e 4 differiscono per la sorgente del segnale. Se differissero
    anche per lo scaler, il confronto misurerebbe due cose insieme e nessuna
    delle due separatamente.
    """
    three = _by_block(3)[0]
    four = _by_block(4)[0]
    assert three.source != four.source
    assert scaler_spec(three) == scaler_spec(four) == (500, True)


def test_block_three_does_not_follow_its_own_preprocessing():
    """Il caso che sembra un errore e non lo e'.

    Il blocco 3 gira a 100 Hz senza notch, come il blocco 0, ma usa lo scaler
    del confronto. Il suo scopo e' isolare l'effetto del notch contro il blocco
    1: cambiare insieme anche la normalizzazione lo confonderebbe.
    """
    three = _by_block(3)[0]
    zero = _by_block(0)[0]
    assert (three.fs, three.notch) == (zero.fs, zero.notch)
    assert scaler_spec(three) != scaler_spec(zero)


def test_git_sha_is_none_outside_a_repository(tmp_path):
    assert git_sha(tmp_path) is None
