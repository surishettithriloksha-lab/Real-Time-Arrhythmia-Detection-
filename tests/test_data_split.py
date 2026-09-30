"""Contracts for patient-independent record partitions."""

from src.data import EXCLUDED_RECORDS, TEST_RECORDS, TRAIN_RECORDS, VAL_RECORDS, split_records


def test_patient_isolated_splits() -> None:
    """Train, validation, and test records must be pairwise disjoint."""
    assert not set(TRAIN_RECORDS) & set(VAL_RECORDS)
    assert not (set(TRAIN_RECORDS) | set(VAL_RECORDS)) & set(TEST_RECORDS)


def test_paced_records_are_excluded() -> None:
    """All paced records are explicitly rejected from usable partitions."""
    all_usable = set(TRAIN_RECORDS) | set(VAL_RECORDS) | set(TEST_RECORDS)
    assert not all_usable & set(EXCLUDED_RECORDS)
    assert split_records()["excluded"] == EXCLUDED_RECORDS
