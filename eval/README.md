# Answer-quality harness

`eval/harness.py` asks the app the breeder questions in `eval/questions.json` and checks each answer against an answer that
code computes from the same saved data. It is how we measure whether the app answers real questions correctly.

## What it does

1. Copies the cache folder you name into a new results folder, `eval/results/<UTC time>/cache`. The original is never
   changed. (`eval/results/` is in `.gitignore`.)
2. Computes each expected answer by code from that copy: from the raw study and trait catalog records, and, for the
   measurement questions, from the cached measurements (on sweetpotatobase.org and cassavabase.org from the cached plot
   records, as the app reads them there: their /observations leaves out values recorded as 0). Nothing is requested. A question whose answer cannot be computed
   from the copy is skipped, never run.
3. Sends each question, as a new chat, through the app's own pipeline (front door, planner, approvals, retrieval, analysis)
   on the copy. A scripted approver types only "approve" at the data request's approval screen and "continue" at the data
   review. It never approves a catalog setup, never answers a question the app asks, never says "yes" to "look it up?", and
   never accepts or rejects an answer: it ends the run at the final review, so the review stays "pending".
4. Writes `results.json` and `results.md`: question, expected, got, pass or fail, model calls, tokens in and out,
   cost, seconds, and how each question ended. Tokens come from the runs' model logs and the front door's replies; cost
   uses the prices in `PRICES` in `harness.py`, by the model each reply names ("?" for a model with no price, "at least"
   when a call had no counts; the app does not keep OpenAI's cached-input count, so its cost is an upper bound).

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
the reply). `--only q01,q13` runs some of them. Use the `part2\cache\live` folder: it holds the catalogs and the /observations page of study 05DISE178HFLNO7 (the
app's `cache\space_mcp` holds only the catalogs). q13 and q14 also need that study's plot records, which the folder did not
hold on 2026-10-22: until they are added, q13 and q14 are skipped.

## Reading the results

- **PASS** means the expected value appears in the answer the person sees: the chat's last message for the question, and
  only its code-written part. What the app marks as model text ("Additional model notes (unverified)" with its bullets,
  and "Unverified model note: ...") is left out, and numbers recorded inside the run (its typed claims) do not count; they
  stay in `results.json` for reading. (Eval baseline 20261016T150241Z: q04 passed on a recorded 22 while the person saw
  "Question blocked"; scored this way, that baseline is 10 of 14.) Only an answer written by code is scored: a run's answer,
  a list, or a catalog answer. A question that ended with the app asking something, offering a lookup, replying in a
  model's words, or showing that a run stopped ("Question blocked.", "Incomplete result." ...) showed no answer and fails.
  Dates and times, such as the catalog's fetch time, are not read as numbers.
  - a count: the number in the answer;
  - a top location or study type: its name followed by its number of studies;
  - studies per study type: every type with its own count (the first number after its name);
  - a mean: the number right after "Mean" (as in "Mean: 2.37857"), rounded at the decimals shown; a whole number counts
    only as the exact value (2 never passes for 2.38).
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
