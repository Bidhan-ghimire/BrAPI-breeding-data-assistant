# Sweetpotato benchmark: single-statistic sub-questions

`spb_subquestions.json` holds the 33 benchmark items computed from SweetPotatoBase plot data (`verification.origin`
starts with `spb_native`) from the sweetpotato breeding benchmark. That benchmark has 140 questions with verified gold
answers and was built 16 to 17 Oct 2026, independently of this app. Its source copy is
`CassavaBase\Claude outputs\benchmark_20261021\sweetpotato_breeding_benchmark.json`; the sha256 is in the file. Each item
is split into sub-questions of one statistic, or one tight group that one statistic gives. That makes 173 sub-questions
for 176 key facts:

| Tier | Meaning | n |
|---|---|---|
| T1 | one study: count, mean, highest or lowest, design metadata | 54 |
| T2 | two or more studies: means, differences, % change, correlation of entry means | 82 |
| T3 | ANOVA, H2, variance components, STI, p-values | 37 |

Eight are marked `checkable: false` (published paper values, or only "about half"). `eval/benchmark.py` lists them and
never runs them.

## Conventions
The file's own `notes` describe these:
- `expected_text`: case-insensitive. "A|B" is satisfied by either name.
- `expected[].abs`: each number's own display tolerance. A count is always exact.
- Reductions are asked as positive percentages; signed quantities keep their sign.
- `alternatives`: values the gold's `grading_notes` also accept.

## Data caveats (the benchmark's README)
Sub-questions that use these studies carry the text in `caveat`:
- MZP2018GUR-ST02 (4576) repeats 61 of 66 plot yields of its irrigated twin (4558).
- MZP2017GUR-ST02 and ST03 share 16 and 31 of 66 yields with ST01.
- MZP2017UMB-ST01 (4788) repeats much of MZP2016UMB-ST01's (4777) dry-matter and beta-carotene data.

The gold is raw means and simple ANOVA from a 16 Oct 2026 snapshot, not checked by a human breeder. When the app and the
gold disagree, check both before calling either wrong. The yield variable is
'Total storage roots yield computation tons per ha|CO_331:0000296' (unadjusted); CO_331:0000999 is a different number.

## How it was made and checked (2026-10-21)
- One agent (Fable) split the items from their question, answer, key facts, grading notes, records and verification
  evidence.
- Checked by script:
  - every key fact of all 33 items is covered exactly once;
  - the source sha256 matches;
  - `benchmark.load_subquestions` accepts all 173;
  - 372 of 382 expected values appear in the item's own text. The other ten are derived or written in words (A42.1
    "seven"/"four", B26.5 and B30.2 study IDs from records, B29.3 "only Chingova", E16.4 MS ratios, E13.6 not
    checkable) or written in scientific notation (A40.4, D14.6).
- A sample was read against the gold.
