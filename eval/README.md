# Answer-quality harness

`eval/harness.py` asks the app the breeder questions in `eval/questions.json` and checks each answer against an answer that
code computes from the same saved data. It is how we measure whether the app answers real questions correctly.

## What it does

1. Copies the cache folder you name into a new results folder, `eval/results/<UTC time>/cache`. The original is never
   changed. (`eval/results/` is in `.gitignore`.)
2. Computes each expected answer by code from that copy: from the raw study and trait catalog records, and, for the
   measurement questions, from the cached measurements. Nothing is requested. A question whose answer cannot be computed
   from the copy is skipped, never run.
3. Sends each question, as a new chat, through the app's own pipeline (front door, planner, approvals, retrieval, analysis)
   on the copy. A scripted approver types only "approve" at the data request's approval screen and "continue" at the data
   review. It never approves a catalog setup, never answers a question the app asks, never says "yes" to "look it up?", and
   never accepts or rejects an answer: it ends the run at the final review, so the review stays "pending".
4. Writes `results.json` and `results.md`: question, expected, got, pass or fail, model calls, seconds, and how each
   question ended.

By default nothing is requested from the database: a request for anything not in the copy is refused and counted
(`--allow-live` lets the app fetch it after the scripted approval, as it would in the chat). The model calls are real:
the app's configured model, from its usual settings.

## Running it (Bidhan's command)

From the repository folder, with the project interpreter:

```
..\CassavaBase\part2\.venv\Scripts\python eval\harness.py --cache "..\CassavaBase\part2\cache\live"
..\CassavaBase\part2\.venv\Scripts\python eval\harness.py --cache "..\CassavaBase\part2\cache\live" --confirm
```

The first line only prints what would run. The second runs it: 14 questions, about 1 front-door model call each and a few
more for each question that starts a run. A question the front door's model labels unsure, out of scope, a greeting or an
app question while the saved catalog is in view costs 2 front-door calls (the label call, then the talk call that writes
the reply). `--only q01,q13` runs some of them. Use the `part2\cache\live` folder: it holds the measurements of study 05DISE178HFLNO7 that q13 and q14 need (the
app's `cache\space_mcp` holds only the catalogs).

## Reading the results

- **PASS** means the expected value appears in the answer as shown, or as a typed claim of the run:
  - a count: the number in the answer, or a count claim with that value;
  - a top location or study type: its name followed by its number of studies;
  - studies per study type: every type with its own count (the first number after its name);
  - a mean: the number the answer gives right after "pooled mean" (or the first "mean"), rounded at the decimals shown
    (an integer alone never passes), or an ungrouped mean claim (a clone's mean does not count).
  Read the "got" column before trusting a pass: a number can still appear for another reason (for example "1 study" in the
  totals line).
- **FAIL** shows why: the value is missing, the app asked a question, or the run stopped at a step the harness never takes.
- An answer of zero said in words ("No study names a season with 2025") is not counted as 0; read it yourself.
- A question that hangs waits as long as the chat would (15 minutes per step). A run the chat stops waiting for is ended by
  the harness; while any run still goes on, database requests stay refused in that process ("runs_still_going" in
  results.json).

## Tests

`tests/test_eval_harness.py` runs the harness end to end offline with scripted models, a fake server and a copy of a
synthetic cache: nothing is sent, no model is called.
