from scripts.baselines.problem2.run_problem2_fair_non_six_queue import jobs


def test_non_six_queue_has_only_forty_two_method_view_seed_jobs():
    planned = jobs()

    assert len(planned) == 42
    assert len({(method, view, seed) for method, view, _, seed in planned}) == 42
    assert {method for method, _, _, _ in planned} == {
        "masked_train", "emoe", "cacr", "tlra", "mruf", "qa_moe", "ebmc",
    }
    assert {view for _, view, _, _ in planned} == {"aligned_po", "unaligned_po"}
    assert {seed for _, _, _, seed in planned} == {1, 2, 3}
