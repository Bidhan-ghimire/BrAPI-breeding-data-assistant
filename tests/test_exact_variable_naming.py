"""P1 and P2 (granted by the coach on 2026-10-22 at 16:50 UTC, as one change): among variables that share the name before the
first "|", one the person names exactly — its whole name with an ontology suffix, or its ID in an answer to our question — wins
over its same-named siblings (P1); and when every matching variable is named exactly, the plan uses all of them (P2). The live smoke check of the benchmark's answer showed why: "Storage root dry matter content
computing percent|CO_331:0000297" also named 78652, 78653, ... ("...|day 1|after harvest|COMP:0000102") by that prefix, so the
planner asked "More than one trait matches ... Which one?" again after the person typed 76605. Other variables are untouched:
a variable named only by its name keeps counting unless one of ITS same-named siblings is named exactly. SYNTHETIC metadata."""
import agents.coordinator as co

DRY = co.KnownVariable(variable_id="76605", name="Storage root dry matter content computing percent|CO_331:0000297")
DRY_DAY1 = co.KnownVariable(variable_id="78652", name="Storage root dry matter content computing percent|day 1|after harvest|COMP:0000102")
DRY_MONTH4 = co.KnownVariable(variable_id="78653", name="Storage root dry matter content computing percent|month 4|after harvest|COMP:0000103")
BETA = co.KnownVariable(variable_id="76619", name="Content of beta-carotene in dry weight basis in raw storage roots measuring mg per 100g|CO_331:0000289")
YIELD = co.KnownVariable(variable_id="V9", name="fresh root yield")                    # a name without an ontology suffix
META = co.KnownMetadata(base_url="https://synthetic.invalid/brapi/v2", complete=True, variables=[DRY, DRY_DAY1, DRY_MONTH4, BETA, YIELD])


def ids(question: str) -> list[str]:
    return [v.variable_id for v in co.find_candidates(question, META).variables]


def test_a_whole_name_with_its_suffix_wins_over_its_same_named_siblings():
    assert ids(f"What is the mean '{DRY.name}' in study 4610?") == ["76605"]


def test_the_id_the_person_typed_wins_over_the_same_named_siblings():
    question = co.add_clarification("What is the mean storage root dry matter content computing percent in study 4610?", "76605")
    assert ids(question) == ["76605"]


def test_without_an_exact_naming_the_siblings_all_stay_and_the_planner_still_asks():
    assert ids("What is the mean storage root dry matter content computing percent in study 4610?") == ["76605", "78652", "78653"]


def test_an_exact_naming_never_removes_a_variable_of_another_name():
    assert ids(f"Mean '{DRY.name}' and fresh root yield in 4610") == ["76605", "V9"]
    assert ids(f"'{DRY.name}' against content of beta-carotene in dry weight basis in raw storage roots measuring mg per 100g") \
        == ["76605", "76619"]


def test_two_variables_named_exactly_both_stay():
    """P1 removes only same-named siblings; the two-trait question is P2's (below)."""
    assert ids(f"Correlate '{DRY.name}' with '{BETA.name}' across trials") == ["76605", "76619"]


# --- P2 (granted with P1, coach 2026-10-22 16:50 UTC): a question that names several variables exactly plans with all of them --

from test_coordinator import completed, draft, fake, plan_with  # noqa: E402

STUDY = co.KnownStudy(study_id="S1", name="synthetic study one")
META_S1 = co.KnownMetadata(base_url=META.base_url, complete=True, studies=[STUDY], variables=[DRY, DRY_DAY1, DRY_MONTH4, BETA, YIELD])
TWO = f"In study S1, what are the mean '{DRY.name}' and the mean '{BETA.name}'?"


def two_trait_draft(variables):
    steps = [{"step_id": f"step_{i}", "agent": "retriever", "action": "get_observations", "inputs": {"study_db_id": "S1", "variable_db_id": v}}
             for i, v in enumerate(variables, 1)]
    steps.append({"step_id": f"step_{len(variables) + 1}", "agent": "analyst", "action": "numeric_summary", "inputs": {},
                  "depends_on": [s["step_id"] for s in steps]})
    return draft(interpretation="descriptive means of the two named traits in study S1", scope={"study_ids": ["S1"], "variable_ids": list(variables)},
                 statistic={"kind": "mean", "denominator": "valid plot-level values", "grouping": []}, steps=steps)


def test_two_variables_named_exactly_are_planned_without_asking_which_one(tmp_path):
    result = plan_with(tmp_path, TWO, fake(completed(two_trait_draft(["76605", "76619"]))), META_S1)
    assert result.status == "completed" and result.plan.status == "ready" and result.questions == []
    assert result.plan.scope.variable_ids == ["76605", "76619"]


def test_a_plan_that_leaves_out_a_variable_named_exactly_is_sent_back(tmp_path):
    model = fake(completed(two_trait_draft(["76605"])), completed(two_trait_draft(["76605", "76619"])))
    result = plan_with(tmp_path, TWO, model, META_S1)
    assert result.status == "completed" and result.attempts == 2 and result.plan.scope.variable_ids == ["76605", "76619"]


def test_which_one_is_still_asked_when_one_variable_is_named_only_by_its_name(tmp_path):
    """Decision (6): the question stays whenever not every matching variable is named exactly."""
    question = f"In study S1, what are the mean '{DRY.name}' and the mean content of beta-carotene in dry weight basis in raw storage roots measuring mg per 100g?"
    result = plan_with(tmp_path, question, fake(completed(two_trait_draft(["76605", "76619"]))), META_S1)
    assert result.status == "needs_clarification" and "More than one trait matches" in result.questions[0]


def test_which_one_is_still_asked_when_nothing_is_named_exactly(tmp_path):
    question = "In study S1, what is the mean storage root dry matter content computing percent?"
    result = plan_with(tmp_path, question, fake(completed(two_trait_draft(["76605"]))), META_S1)
    assert result.status == "needs_clarification" and "More than one trait matches" in result.questions[0]



# --- the review of 2026-10-22 (REVIEW.md): guards for its findings, real (E14.4) and made-up (A-E) -------------------------

def ids_in(metadata, question):
    return [v.variable_id for v in co.find_candidates(question, metadata).variables]


def test_an_id_counts_as_exact_only_in_an_answer_to_our_question():
    """Review A: a number in the question that equals a variable ID ("76605 plots") made P1 drop the siblings silently."""
    question = "In study 4610, 76605 plots measured storage root dry matter content computing percent. What is the mean?"
    assert ids(question) == ["76605", "78652", "78653"]                                         # asked, as before P1
    assert ids(co.add_clarification(question, "76605")) == ["76605"]                             # the person's answer decides


def test_small_numeric_ids_in_a_question_are_never_exact(tmp_path):
    """Review A: on a server with IDs 1, 2, 3, "3 reps in 2 trials" made P2 plan with those variables without asking."""
    small = co.KnownMetadata(base_url=META.base_url, complete=True, studies=[STUDY],
                             variables=[co.KnownVariable(variable_id=str(n), name=f"fresh root yield|CO_1:000000{n}") for n in (1, 2, 3)])
    question = "In study S1, what is the mean fresh root yield of 3 reps in 2 trials?"
    result = plan_with(tmp_path, question, fake(completed(two_trait_draft(["1"]))), small)
    assert result.status == "needs_clarification" and "More than one trait matches" in result.questions[0]


def test_names_without_a_suffix_are_never_exact(tmp_path):
    """Review B and E: "fresh root yield" inside "fresh root yield per plant", and a suffixless "starch content" next to
    "starch content|CO_1:9", were both taken as exact. A name without an ontology suffix is no more exact than its base."""
    plain = co.KnownMetadata(base_url=META.base_url, complete=True, studies=[STUDY], variables=[
        co.KnownVariable(variable_id="V1", name="fresh root yield"), co.KnownVariable(variable_id="V2", name="fresh root yield per plant"),
        co.KnownVariable(variable_id="V3", name="starch content"), co.KnownVariable(variable_id="V4", name="starch content|CO_1:0000009")])
    assert ids_in(plain, "In S1, the mean starch content?") == ["V3", "V4"]
    result = plan_with(tmp_path, "In study S1, what is the mean fresh root yield per plant?", fake(completed(two_trait_draft(["V2"]))), plain)
    assert result.status == "needs_clarification" and "More than one trait matches" in result.questions[0]


def test_a_whole_name_inside_a_longer_named_one_is_not_exact(tmp_path):
    inner = co.KnownMetadata(base_url=META.base_url, complete=True, studies=[STUDY], variables=[
        co.KnownVariable(variable_id="V5", name="dry matter|CO_1:0000010"), co.KnownVariable(variable_id="V6", name="root dry matter|CO_1:0000010")])
    result = plan_with(tmp_path, "In study S1, what is the mean root dry matter|CO_1:0000010?", fake(completed(two_trait_draft(["V6"]))), inner)
    assert result.status == "needs_clarification" and "More than one trait matches" in result.questions[0]


def test_two_variables_with_one_whole_name_are_never_exact(tmp_path):
    """Review C: two IDs with the same whole name cannot be told apart by that name."""
    twins = co.KnownMetadata(base_url=META.base_url, complete=True, studies=[STUDY], variables=[
        co.KnownVariable(variable_id="V7", name="dry matter|CO_1:0000001"), co.KnownVariable(variable_id="V8", name="dry matter|CO_1:0000001")])
    result = plan_with(tmp_path, "In study S1, what is the mean dry matter|CO_1:0000001?", fake(completed(two_trait_draft(["V7", "V8"]))), twins)
    assert result.status == "needs_clarification" and "More than one trait matches" in result.questions[0]


def test_a_term_the_candidates_do_not_carry_keeps_the_question(tmp_path):
    """Review, real data (E14.4): iron was named only by its term ("SPB term CO_331:0001016 ... or CO_331:0000685"), so it was no
    candidate; with every other candidate exact, P2 planned without iron and asked nothing. Now an ontology term in the question
    that no candidate carries keeps the "Which one?" question, as before P2."""
    question = f"In study S1, the mean '{DRY.name}', the mean '{BETA.name}' and iron (SPB term CO_331:0001016 or CO_331:0000685)?"
    result = plan_with(tmp_path, question, fake(completed(two_trait_draft(["76605", "76619"]))), META_S1)
    assert result.status == "needs_clarification" and "More than one trait matches" in result.questions[0]


def test_three_variables_named_exactly_are_all_planned(tmp_path):
    third = co.KnownVariable(variable_id="76581", name="Total storage roots yield computation tons per ha|CO_331:0000296")
    meta = co.KnownMetadata(base_url=META.base_url, complete=True, studies=[STUDY], variables=[*META_S1.variables, third])
    question = f"In study S1, the means of '{DRY.name}', '{BETA.name}' and '{third.name}'?"
    result = plan_with(tmp_path, question, fake(completed(two_trait_draft(["76605", "76619", "76581"]))), meta)
    assert result.status == "completed" and result.plan.status == "ready" and result.plan.scope.variable_ids == ["76605", "76619", "76581"]


def test_a_plan_that_leaves_out_the_first_variable_is_sent_back(tmp_path):
    model = fake(completed(two_trait_draft(["76619"])), completed(two_trait_draft(["76605", "76619"])))
    result = plan_with(tmp_path, TWO, model, META_S1)
    assert result.status == "completed" and result.attempts == 2 and result.plan.scope.variable_ids == ["76605", "76619"]


def test_an_id_in_an_answer_matches_as_a_whole_word_in_any_case():
    question = "What is the mean storage root dry matter content computing percent in study 4610?"
    assert ids(co.add_clarification(question, "776605")) == ["76605", "78652", "78653"]           # not 76605
    meta_v = co.KnownMetadata(base_url=META.base_url, complete=True, variables=[
        co.KnownVariable(variable_id="V9", name="dry matter|CO_1:0000001"), co.KnownVariable(variable_id="V10", name="dry matter|day 1|COMP:0000002")])
    assert ids_in(meta_v, co.add_clarification("mean dry matter in S1?", "v9")) == ["V9"]


def test_the_planner_note_says_what_the_plan_must_hold(tmp_path):
    result = plan_with(tmp_path, TWO, fake(completed(two_trait_draft(["76605", "76619"]))), META_S1)
    assert any("names ['76605', '76619'] exactly; all of them are required in the plan" in n for n in result.notes)


def test_a_whole_name_glued_to_more_characters_is_not_exact():
    """"dry matter|CO_1:0000001" inside "dry matter|CO_1:00000012" (another term) names nothing exactly; the siblings stay."""
    glued = co.KnownMetadata(base_url=META.base_url, complete=True, variables=[
        co.KnownVariable(variable_id="V11", name="dry matter|CO_1:0000001"), co.KnownVariable(variable_id="V12", name="dry matter|day 1|COMP:0000002")])
    assert ids_in(glued, "In S1, the mean dry matter|CO_1:00000012?") == ["V11", "V12"]
