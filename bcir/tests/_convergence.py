"""The shared E-demo convergence gate (the H4 follow-on): a training run "demonstrates convergence" only when
(a) the loss DECREASED, (b) the final loss clears an ABSOLUTE threshold, (c) the drop is substantial
(final <= max_ratio * initial), and (d) where a frozen-baseline ceiling is supplied, the trained model beats
it. This keeps "the loss went down a bit" from ever passing as a convergence demo (the audit finding: the E6
gate required only a 2x drop while the measured run achieved 70x). Test-rail helper only -- no verifier
imports, no Diagnostic, nothing graded (the two-truth quarantine)."""


def assert_converged(train_loss, *, final_below, max_ratio=0.5, baseline=None, name="training run"):
    """Gate a per-epoch loss history as a genuine convergence demonstration."""
    assert train_loss, f"{name}: empty loss history"
    first, last = train_loss[0], train_loss[-1]
    assert last < first, f"{name}: loss did not decrease ({first} -> {last})"
    assert last < final_below, f"{name}: final loss {last} misses the absolute gate {final_below}"
    assert last <= max_ratio * first, (
        f"{name}: drop not substantial ({first} -> {last}, ratio {last / first:.4f} > {max_ratio})"
    )
    if baseline is not None:
        assert last < baseline, (
            f"{name}: trained loss {last} does not beat the frozen-baseline ceiling {baseline}"
        )
