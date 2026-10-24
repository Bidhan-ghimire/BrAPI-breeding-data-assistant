"""(28)(ii), Bidhan's decision (coach 2026-10-24 17:25 UTC; Luna A42.4): the Analyst reads each tool's docstring as its
description. A42.4's Analyst had the 315 genotype means from group_stats and asked for "a further numeric calculation tool",
though numeric_summary on the group_stats result is that tool. Both descriptions now say so, and that the saved tables keep
every group; the calls they name give the mean of all the saved group means, each group weighted equally, and filter_rows
leaves groups out (review 2026-10-24). Everything here is SYNTHETIC."""
import statistics

import analyst_tools as at
from test_correlate_entry_means import claims, registry, row, save_rows  # noqa: F401  (the registry fixture)
from test_view_says_what_the_table_is import group_table, plots, run


def mean_claim(result):
    return next(c for c in claims(result) if c.kind == "mean")


def test_the_descriptions_say_how_to_take_a_mean_of_group_means():
    groups = at.ANALYST_TOOLS["group_stats"].description
    summary = at.ANALYST_TOOLS["numeric_summary"].description
    assert 'call numeric_summary on the result_artifact_id with value="mean"' in groups
    assert "the result keeps every group, included or not, so filter_rows on it first" in groups
    assert 'value="mean" gives the mean of the group means, each group weighted equally' in summary
    assert "Those tables keep every group: included=false" in summary and "left_out=true" in summary


def test_the_call_the_descriptions_name_gives_the_mean_of_the_group_means(registry):
    """plots(): A 10 and 12, B 20 and 22, C 30 and 32, D 40 and 42, K 50 and 52: group means 11, 21, 31, 41, 51; their mean 31."""
    mean = mean_claim(run(registry, "numeric_summary", artifact_id=group_table(registry, plots(registry)), value="mean"))
    assert mean.value == 31.0 and mean.denominator.n == 5


def test_every_saved_group_counts_once_whatever_its_plots(registry):
    """Review 2026-10-24: a balanced table of 5 groups could not tell the mean of the group means from the plot mean, a median,
    or the top_n groups the message shows. 12 groups (top_n is 10) with 1 to 4 plots each: group i holds 10i, 10i+1, 10i+4,
    10i+9 (a mean that is not the median)."""
    rows = [row(f"u{i:02d}{j}", f"G{i:02d}", f"SYNTHETIC G{i:02d}", str(10 * i + j * j), obs=f"o{i:02d}{j}") for i in range(12) for j in range(1 + i % 4)]
    groups = run(registry, "group_stats", artifact_id=save_rows(registry, rows, ["S1"]), by="germplasmDbId", value="value",
                 min_independent_n=1).data["result_artifact_id"]
    mean = mean_claim(run(registry, "numeric_summary", artifact_id=groups, value="mean"))
    expected = statistics.fmean(statistics.fmean(10 * i + j * j for j in range(1 + i % 4)) for i in range(12))
    medians = statistics.fmean(statistics.median(10 * i + j * j for j in range(1 + i % 4)) for i in range(12))
    assert mean.denominator.n == 12 and abs(mean.value - expected) < 1e-9 and abs(expected - medians) > 0.4


def test_an_entry_rank_entries_left_out_counts_unless_filtered_as_the_description_says(registry):
    ranked = run(registry, "rank_entries", artifact_id=plots(registry), by="germplasmDbId", value="value", exclude=["K"]).data["result_artifact_id"]
    every = mean_claim(run(registry, "numeric_summary", artifact_id=ranked, value="mean"))
    kept = run(registry, "filter_rows", artifact_id=ranked, column="left_out", op="eq", value="false").artifact_ids[0]
    without = mean_claim(run(registry, "numeric_summary", artifact_id=kept, value="mean"))
    assert (every.value, every.denominator.n) == (31.0, 5) and (without.value, without.denominator.n) == (26.0, 4)
