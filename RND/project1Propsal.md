# Project 1 — Fresher Friend Matching I dont have a name to the project yet

**Committee:** AI/ML · **Team size:** 9 · **Duration:** 14 weeks
**This is the PoC project.** 
Mathematics: contains reasonable assimptions and will be added in the appropriate section.

---

## 1. Project Overview

### 1.1 Problem statement

A first-year student arrives on a campus of roughly 5000 strangers. The social graph they end up
with is determined almost entirely by allocation accidents : which room, which section, who sat
down first in the mess. Students who are shy, who joined late, or whose interests are unusual end
up structurally isolated, and no existing mechanism corrects for that.

The obvious response : build a friend recommender which walks into a specific and severe constraint,
which is the real problem statement:

> **The system will observe approximately 12 verified friendship outcomes in a semester.** yep ik bold claim but trust me i mean it when I say i made reasonable assumptions and even conducted surveys from the upcoming freshers.
we assume 500 ppl ocme to the server 
Registration -> onboarding completion -> impression -> accept -> real conversation -> durable
friendship at 60 days -> survey response is six multiplicative stages. At optimistic rates, 500
registrations produce ~1,500 impressions and ~12 confirmed outcomes. Cheap proxy labels are
abundant; true labels are almost absent. 

Building a recommender that can be *checked* on 12 labels — and knowing
which techniques that budget forbids — is the actual technical problem.

### 1.2 Objectives

1. Establish the label budget by measurement, replacing the assumed funnel rates with the club's
   own numbers.
2. Build a **simulation and evaluation harness first**, capable of scoring an assignment before
   any real user registers.
3. Build the scoring pipeline: soft-IDF interest similarity, causal-uplift co-location, activity
   and schedule overlap, with hand-set weights justified by published effect sizes.
4. Build the assignment: a monotone submodular objective over a degree-constrained subgraph, with
   a proved approximation guarantee and a proved feasibility guarantee.
5. Instrument the serving path with **randomised exploration and propensity logging** from the
   first impression served.
6. Run one adequately powered A/B test and report it honestly, including what it could not resolve.
7. Operate a **recurring weekly activity slot**, which the contact-hour arithmetic identifies as
   the highest-value component in the entire system.

### 1.3 Expected impact

**For the campus.** A measurable reduction in the number of first-years with no recommended
contacts, and a recurring activity that accrues the co-present hours friendship actually requires.

**For the club.** Members finish able to compute a label budget before proposing a model, which is
a skill most working which most people only get after the Model crashes on low metrics and there is confusion. The evaluation
harness outlives the semester: next year's team inherits a measurement apparatus, not a codebase to
rewrite.

**For the applicant pool.** The appendix is a reusable template — a document where every
architectural decision names the number that forced it.

### 1.4 Why it is worth pursuing

Three properties, and the project was chosen because it has all three:

- **The failure of the naive approach is computable in advance.** This one rests on a power calculation that a second-year can run in
  an afternoon.
- **It is honest about a situation ML students rarely face.** Every course dataset has labels. This
  one does not, and learning to recognise that condition — and what remains possible inside it — is
  more transferable than another CNN. 
- **Nothing depends on permission.** No GPU, no paid API, no data-access request to any authority
  who might say no. The riskiest external dependency is booking a room.

---

## 2. Technical Plan

### 2.1 Proposed solution

Five stages, described fully in [`../poc/architecture.md`](../poc/architecture.md):

1. **Ingestion.** Onboarding captures interests, free schedule slots, branch , and an 8–12
   item scenario game. The game exists because onboarding completion is the highest-elasticity
   variable in the funnel, well not because its answers are a good ranking signal (they are not; see
   §2.3).
2. **Scoring.** A sentence encoder embeds ~1,000 unique interest strings. Soft document frequency
   replaces exact counting so that rare, informative interests remain matchable. MaxSim gives
   asymmetric coverage; the harmonic mean symmetrises it and punishes one-sided matches. Six
   features combine into $S_{uv} = \sum_k \beta_k f_k$ with **hand-set** $\beta$.
3. **Assignment.** A single monotone submodular objective — pair score plus attribute coverage —
   maximised greedily over the degree-constrained family with cap $c=5$ and floor $\ge 1$.
4. **Serving.** 80% exploit, 20% explore over the mid-score band, with the propensity logged at
   serve time. Matched groups are invited to a recurring weekly activity.
5. **Evaluation.** Blocking-pair audit, SNIPS off-policy estimates, one A/B test, and univariate
   contrasts feeding manual weight revision.

**What is deliberately absent, by calculation:** no training loop, no model registry, no feature
store, no vector database, no ANN index, no candidate-generation stage.

### 2.2 Technologies and tools

| Layer | Choice | Reason |
|---|---|---|
| Embeddings | `sentence-transformers`, MiniLM-class, $d = 384$, CPU | 1,000 strings encode in seconds. GPU is unjustifiable at this scale. |
| Scoring | NumPy | The full similarity matrix is one $10^6 \times 384$ matmul — 384 MFLOP, milliseconds. |
| Assignment | Python + `heapq` for lazy greedy | 44,850 pairs. A sort and a sweep. |
| Exact reference | PuLP / OR-Tools | Integer-programming solve at $n \le 40$ to measure the realised approximation ratio against the guarantee. |
| Front end | Discord bot, or a minimal web form | Whichever the cohort already uses. This is not the interesting part and should not consume the interesting people. |
| Storage | SQLite | 300 users. Anything else is theatre. |
| Logging | Append-only JSONL, schema-validated | Propensity is a required field. A dropped field is unrecoverable. |
| Analysis | `scipy.stats`, `statsmodels` | Power calculations, Fisher $z$, two-proportion tests. |

### 2.3 Expected challenges

| Challenge | Why it is hard | Mitigation |
|---|---|---|
| **The encoder may not separate the vocabulary** | Sentence embeddings are anisotropic : random pairs have positive expected cosine ($\mu_0 \approx 0.2$–0.4). If the optimal threshold sits inside $\mu_0 + \sigma_0$, no threshold rescues it. | Explicit **acceptance test in week 4**: require $\tau^\star > \mu_0 + \sigma_0$ on 200 hand-labelled pairs. Fail it and change the encoder before building anything downstream. |
| **The funnel estimate may be wrong** | Six assumed rates, multiplied. The conclusion is robust across two orders of magnitude, but the number 12 is not defended. | Week 1 replaces the assumed $\eta_1$ with the club's measured onboarding completion. If gold labels land near 40, the parameter budget changes and §13 of the appendix is rewritten. **That is the document behaving correctly.** |
| **Cold start on activities** | $f_4$ (activity overlap) carries the largest weight and requires activities to exist. | The recurring-slot track starts in **week 3, in parallel**, and does not wait for the matcher. |
| **Nobody wants to build the evaluation harness** | It is the least glamorous component and the first deliverable. | It is staffed first and it is the only sub-team whose week-1 output every other sub-team depends on by week 7. Making it upstream makes it visible. |
| **Safety and reporting** | The system introduces strangers to each other. | Block/report in the first shipped version, a named human owner and well a offer letter. 
| **The A/B test will probably be inconclusive** | Powered to 20 percentage points only; most realistic changes move less. | Pre-register the effect size in week 11. Commit in advance to reading a null as *uninformative*, because at this power failure to reject carries no evidence either way. |
| **Segregation hazard** | Region and language predict friendship, and weighting them reproduces existing segregation. | $\beta_5$ capped at 0.05, and the cap is documented as a **value judgement, not a mathematical result**. |

### 2.4 Roadmap with key milestones

Full version with gates: [`../poc/roadmap.md`](../poc/roadmap.md).

| Weeks | Milestone | Gate to pass |
|---|---|---|
| 1–2 | Simulation harness; synthetic preference profiles; blocking-pair evaluator; exact IP reference | $B(X)$ computable at $n = 40$ with a brute-force cross-check |
| 3–4 | Encoder evaluation; $\mu_0, \sigma_0$ measured; $\tau^\star$ calibrated on 200 labelled pairs | **Hard gate:** $\tau^\star > \mu_0 + \sigma_0$, or change the encoder |
| 3–14 | *(Parallel track)* Recurring activity slot: venue, named hosts, calendar | One slot running weekly by week 5, independent of the matcher |
| 5–6 | Scoring pipeline; soft-IDF; $f_1$–$f_6$; full 44,850-pair matrix | Full recompute under 1 s; feature distributions inspected, not assumed |
| 7–8 | Assignment; lazy greedy; repair pass; $\lambda$ sweep | Welfare-vs-stability frontier plotted; realised ratio measured against the 1/3 bound |
| 9–10 | Onboarding flow; scenario game; front end | Onboarding completion measured — the highest-elasticity variable in the funnel |
| 11 | **Logging and exploration arm** | Propensity captured at serve time. **No launch without this.** |
| 12 | Pilot: 40–60 users, one hall | Round trip works; logs complete; zero safety incidents |
| 13–14 | A/B test #1; univariate contrasts; report | Report states what was *not* learned, at the resolution power permits |

---

## 3. Team and Learning Plan

### 3.1 Structure and delegation (9 members)

| Sub-team | Size | Owns | Week-1 task | How it self-checks |
|---|---|---|---|---|
| **Simulation & evaluation** | 2 | Harness, blocking-pair metric | Generate synthetic preference profiles; count blocking pairs in a random assignment | Brute-force enumeration at $n = 8$ |
| **Scoring & representation** | 2 | Encoder, soft-IDF, MaxSim, $\tau$ calibration | Encode 1,000 interest strings; plot the cosine histogram; report $\mu_0, \sigma_0$ | The histogram either separates or it doesn't — visible immediately |
| **Assignment & optimisation** | 2 | Greedy sweep, repair pass, $\lambda$ sweep | Greedy $b$-matching on a 40-node synthetic graph | Compare against exhaustive search at $n = 10$ |
| **Product & operations** | 2 | Onboarding, game, **recurring activity slot** | Run one activity session; measure show-up rate $\phi$ | $\phi$ is a count over a count |
| **Experimentation & logging** | 1 | Impression schema, propensity capture, SNIPS, A/B design | Write the schema; document what becomes unrecoverable if each field is dropped | Reviewed by the whole team — a missing field is a permanent loss |

**Why this split.** Each sub-team owns a stage of the pipeline end to end, so nobody is a
"frontend person" who never touches the argument. **Nothing blocks until week 5**, so a team that
stalls in week 2 has stalled alone.

### 3.2 Onboarding and mentoring beginners

**The design constraint on every week-1 task was: it must be checkable without a mentor.** A
first-year runs one command and knows within seconds whether the output is right — against a
brute-force enumeration, a published figure, or a histogram they can look at.

**The second constraint: a beginner's first contribution should be a measurement that replaces an
assumption.** The appendix is full of labelled assumptions. Week 1 for the scoring team measures
$\mu_0$; week 1 for operations measures $\phi$; week 1 for evaluation measures $B(X_{\text{rand}})$.
Each one deletes a "*(assumption)*" tag from a document the whole club can see. That is a much
better first week than closing a starter issue.

**Mentoring model.** Each team has one member who has shipped something before, paired with one
who has not. Weekly 30-minute cross-team demos where each sub-team explains *one number they
produced* — not code walkthroughs. The demo format enforces the project's own thesis.

### 3.3 Workshops required

| # | Week | Title | Content |
|---|---|---|---|
| 1 | 1  Reproducing the funnel on the club's own numbers. Elasticity: why all six stages have equal leverage and you prioritise by cost to move. |
| 2 | 3 | Embeddings and the anisotropy problem | Cosine similarity is not zero for unrelated text. Taught over the histogram the team generated in their own week 1. |
| 3 | 6 | Information content and IDF | Shannon surprisal, $-\ln p$, and why IDF is not an arbitrary weight. |
| 4 | 7 | Submodularity and greedy guarantees | Diminishing returns, facility location, and a whiteboard proof that degree constraints form a 2-extendible system. Why the constant is 1/3 and not $1-1/e$. |
| 5 | 9 | Stable matching and non-existence | Gale–Shapley, then Irving. The four-person counterexample where no stable matching exists. |
| 6 | 11 | Causal inference and why we randomise | Uplift vs observational probability. Propensity logging as the thing that cannot be retrofitted. |
| 7 | 13 | **How to report a null result honestly** | Power, minimum detectable effect, and why failure to reject at $n = 12$ is not evidence of absence. |

### 3.4 Resources required

No GPU. No paid API. No cloud budget. Nine laptops, a room for the recurring activity, a Discord
server or a static web host, and a named faculty or senior contact for the safety escalation path.

Reading: Nemhauser–Wolsey–Fisher (1978) §1, Khattab & Zaharia (2020) §3, Irving (1985), and the
project's own appendix.

---

## 4. Success Metrics

### 4.1 Technical — objective and self-grading

- **Realised approximation ratio** against the exact IP solve at $n \le 40$, compared to the 1/3
  worst-case bound. Measures whether the guarantee is loose in practice.
- **Blocking-pair ratio** $B(X)/B(X_{\text{rand}})$. A mathematically defined quality measure
  computable before any real user registers.
- **Full score recompute wall clock**, against the predicted sub-second budget.
- **Encoder acceptance test** passed or explicitly failed, with $\mu_0$, $\sigma_0$, $\tau^\star$
  and $F_1$ reported together.

### 4.2 Product

- **Onboarding completion rate.** The highest-elasticity variable in the funnel, so the single most
  valuable number the project produces.
- Accept rate, with its confidence interval — never as a bare point estimate.
- Show-up rate $\phi$ at the recurring slot.
- **Number of users with degree zero, which must be zero.** Every aggregate metric will look
  healthy while this fails, which is exactly why it is stated separately.

### 4.3 Member participation

- Week-1 completion across all nine. The best early predictor of whether the team survives, and
  measurable on day seven.
- Members who have shipped in a sub-team other than their own by week 14. Target $\ge 4$.
- Attendance at the recurring activity by team members — the operations track fails silently
  otherwise.

### 4.4 Learning

- **Number of members who can independently derive why 12 gold labels forbid fitting the six
  weights.** Assessed in week 12 by asking, not by quizzing.
- **Number of appendix assumptions replaced by a member's own measurement.** Target $\ge 6$ of the
  labelled assumptions. This is the metric I would weight most heavily, because it measures whether
  the document became the team's rather than mine.
- Members who can explain why the greedy constant is 1/3 here and $1-1/e$ under a cardinality
  constraint.

### 4.5 Documentation quality

- One decision record per irreversible choice, each carrying the number that forced it. Template in
  [`../docs/decisions/`](../docs/decisions/).
- A one-command reproduction path for the evaluation harness.
- A final report that states **what was not established** and why, at the resolution the power
  calculations permit. A report claiming more than 12 labels can support is a failed report
  regardless of what the product did.

### 4.6 Technical innovation

- The welfare-versus-stability frontier for this problem is not, as far as I can find, published
  for a friend-recommendation setting. Producing it is a genuine, if small, contribution.
- The evaluation harness is reusable by any subsequent matching project in the club.

### 4.7 Community impact

- First-years with at least one durable contact who would not otherwise have met — measured, with
  the acknowledgement that at $n_{\text{gold}} = 12$ this is an anecdote and not an estimate.
- The recurring activity slot continues after the semester, independent of the software.
