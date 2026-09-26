from scripts.baselines.problem2.run_problem2_fair_six_baselines_queue import jobs


def test_six_baseline_queue_has_one_job_for_every_method_view_and_seed():
    planned = jobs()

    assert len(planned) == 36
    assert len({(method, view, seed) for method, view, _, seed in planned}) == 36
    assert {method for method, _, _, _ in planned} == {
        "concat_mlp", "early_fusion_gru", "ef_lstm", "mult", "p_rmf", "cmad",
    }
    assert {view for _, view, _, _ in planned} == {"aligned_po", "unaligned_po"}
    assert {seed for _, _, _, seed in planned} == {1, 2, 3}
