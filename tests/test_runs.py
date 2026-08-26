"""La matrice sperimentale contro PROTOCOL.md §6.7.

Questi test non guardano il codice: guardano se il repo sta per eseguire
esattamente i run che ha dichiarato. Sono i numeri della tabella di §6.7,
riscritti qui in modo che un'aggiunta silenziosa di celle fallisca.
"""

from pathlib import Path

import pytest

from ecgres import runs as R

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "configs" / "runs.csv"

#: La tabella di §6.7, trascritta a mano. Se cambia il protocollo, cambia qui,
#: e il cambiamento e' visibile nel diff invece di essere implicito nel codice.
EXPECTED_RUNS_PER_BLOCK = {0: 3, 1: 20, 2: 15, 3: 5, 4: 5}
EXPECTED_TOTAL = 48

#: Tre run del blocco 3 ripetono una configurazione del blocco 0 (§10 voce 14).
EXPECTED_DUPLICATE_PAIRS = 3


def test_block_sizes_match_the_protocol():
    got = {}
    for run in R.enumerate_runs():
        got[run.block] = got.get(run.block, 0) + 1
    assert got == EXPECTED_RUNS_PER_BLOCK
    assert sum(got.values()) == EXPECTED_TOTAL


def test_run_ids_are_unique():
    ids = [r.run_id for r in R.enumerate_runs()]
    assert len(set(ids)) == len(ids)


def test_blocks_zero_and_three_duplicate_three_configurations():
    """La duplicazione della voce 14 e' voluta e deve restare esatta.

    Se un giorno le due configurazioni divergessero — un notch, un rate, un
    seed — le coppie smetterebbero di essere confrontabili e la verifica di
    riproducibilita' su cui poggia la strategia spot instance sparirebbe senza
    fallire. Questo test la tiene in vita.
    """
    runs = R.enumerate_runs()
    duplicates = R.duplicate_configs(runs)
    assert len(duplicates) == EXPECTED_DUPLICATE_PAIRS

    for key, rows in duplicates.items():
        assert sorted(r.block for r in rows) == [0, 3]
        assert sorted(r.selection for r in rows) == ["macro_all", "macro_clean"]
        assert {r.config_key for r in rows} == {key}
        # 100 Hz, Arm A, senza notch, da records500: il blocco 4 condivide i
        # primi tre campi ma non la sorgente, quindi non entra qui.
        assert key[:4] == (100, "A", False, "hr")

    seeds = sorted(key[4] for key in duplicates)
    assert seeds == sorted(R.BLOCKING_SEEDS)


def test_selection_does_not_belong_to_the_config_key():
    """Due run con la stessa ``config_key`` devono produrre gli stessi pesi.

    Se ``selection`` entrasse nella chiave, le coppie del test precedente
    smetterebbero di essere duplicati e il confronto perderebbe senso.
    """
    a, b = (r for r in R.enumerate_runs() if r.fs == 100 and r.seed == 0
            and r.arm == "A" and not r.notch and r.source == "hr")
    assert a.selection != b.selection
    assert a.config_key == b.config_key


def test_selection_metric_follows_deviation_13():
    """`macro_all` solo per la condizione bloccante, `macro_clean` altrove."""
    for run in R.enumerate_runs():
        expected = "macro_all" if run.block == 0 else "macro_clean"
        assert run.selection == expected, run


def test_notch_is_off_only_where_the_protocol_says():
    """Notch presente solo nei blocchi del confronto (1 e 2).

    Assente nel blocco 0 (§6.6), nel 3 (§8.3.4) e nel 4, dove non e' nemmeno
    applicabile: ``records100`` ha la fondamentale di rete sul proprio Nyquist.
    """
    for run in R.enumerate_runs():
        assert run.notch == (run.block in (1, 2)), run


def test_records100_is_used_only_by_block_four():
    """§4: ``records500`` e' la sorgente unica, salvo l'analisi §8.3.2."""
    for run in R.enumerate_runs():
        assert run.source == ("lr" if run.block == 4 else "hr"), run
        if run.source == "lr":
            assert (run.fs, run.notch) == (100, False), run


def test_block_four_differs_from_block_three_only_in_provenance():
    """Il confronto di §8.3.2 deve avere un solo grado di liberta'."""
    three = {r.seed: r for r in R.enumerate_runs() if r.block == 3}
    four = {r.seed: r for r in R.enumerate_runs() if r.block == 4}
    assert set(three) == set(four)
    for seed in three:
        a, b = three[seed], four[seed]
        assert (a.fs, a.arm, a.notch, a.selection) == (b.fs, b.arm, b.notch, b.selection)
        assert a.source != b.source
        # E proprio per questo non sono duplicati: ``source`` sta nella chiave.
        assert a.config_key != b.config_key


def test_arm_b_skips_100hz():
    """A 100 Hz i due bracci coincidono per costruzione (§6.3): non si corre due volte."""
    b_rates = {r.fs for r in R.enumerate_runs() if r.arm == "B"}
    assert b_rates == set(R.ARM_B_RATES)
    assert 100 not in b_rates


def test_five_seeds_per_cell_three_for_the_blocking_check():
    runs = R.enumerate_runs()
    cells: dict[tuple, set[int]] = {}
    for run in runs:
        cells.setdefault((run.block, run.fs, run.arm), set()).add(run.seed)
    for (block, _, _), seeds in cells.items():
        expected = set(R.BLOCKING_SEEDS) if block == 0 else set(R.SEEDS)
        assert seeds == expected


def test_rates_agree_with_the_data_module():
    """``runs`` duplica ``RATES`` per restare senza dipendenze. Qui si vincola."""
    pytest.importorskip("pandas")
    pytest.importorskip("scipy")
    from ecgres.data import RATES

    assert R.RATES == RATES
    assert set(R.ARM_B_RATES) < set(RATES)


def test_committed_csv_matches_the_code():
    """Il CSV e' generato, non scritto a mano: se divergono, vince il codice."""
    assert CSV.exists(), "manca configs/runs.csv: lanciare scripts/make_runs.py"
    assert R.load_runs(CSV) == R.enumerate_runs()


def test_csv_roundtrip_preserves_types(tmp_path):
    path = R.write_runs(tmp_path / "runs.csv")
    assert R.load_runs(path) == R.enumerate_runs()
    assert "\r\n" not in path.read_bytes().decode()


def test_load_rejects_a_different_header(tmp_path):
    path = tmp_path / "runs.csv"
    path.write_text("run_id,block\nfoo,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="intestazione"):
        R.load_runs(path)
