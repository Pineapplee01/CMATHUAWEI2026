from scripts.baselines.problem2.run_aligned_test252_queue import jobs


def test_test252_queue_has_six_methods_and_three_seeds():
    planned = jobs()

    assert len(planned) == 18
    assert {method for method, _, _ in planned} == {
        "concat_mlp", "early_fusion_gru", "ef_lstm", "mult", "p_rmf", "cmad",
    }
    assert {seed for _, seed, _ in planned} == {1, 2, 3}
