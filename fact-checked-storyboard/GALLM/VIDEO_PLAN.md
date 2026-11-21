# Video plan: automated patient videos (version 1)

Written 21 Nov 2026, updated the same evening and again after research R1–R3 · status: **approved by Diya, 21 Nov 2026**
· still open: which heart topics the graph covers (section 10), and 3 label texts (section 7).

## 1. Goal

A person asks any heart question → the system answers with a short video (30–60 s): an **animated cartoon doctor explains
to a cartoon patient**, made by code with no person working on the video. Every spoken word, caption and object on screen
comes from an approved list. Any failed check stops the video. It answers what was asked and always ends with the
care-team reminder.

## 2. What is decided

| Question | Answer |
|---|---|
| Style | Animated 2D cartoon (realistic or 3D avatar: optional, later) |
| How the pictures are made | An **actor kit written once in code** (`GALLM/video_kit/kit.js`); every video reuses it. **Look approved 21 Nov.** |
| Characters | A woman doctor (white coat, stethoscope) and an older man patient (glasses, cardigan) |
| Objects | 18 objects drawn in the kit: no letters, numbers or logos (includes an ICD device and a plain cigarette pack) |
| Language | English now; German later |
| Use | Published once done |
| Clinician check | None. The label says so honestly |
| Topics | Any heart question the graph has guidelines for. No guideline → "please ask your care team" |
| Personal questions | Answered with general guideline facts, plus the closing reminder |
| Delivery | A **library** of reviewed videos, plus **live** videos for new questions |
| Live videos | Shown at once with an "automatic, not yet reviewed" label; a person reviews them later |
| Reusing a library video | Only if the new question finds the **same graph concepts** and the **same facts** (from Part 1) **and** passes a similarity threshold tuned on tricky question pairs (e.g. "ICD replaced" vs "ICD implanted"). Otherwise a new live video is made. The wrong-match rate is measured before live mode goes public (R3) |
| Before going public | A person watches every video and approves or rejects it |
| How people judge the videos | **PEMAT-A/V** (AHRQ) by two raters, after practising together on 3 videos; agreement (κ) reported; 3 items count as "not applicable" for videos under 1 minute. Plus a small **viewer test**: a few lay viewers say the main point in their own words and name the action asked for. PEMAT does not check medical accuracy; our automatic checks do (R2) |
| Test questions | 44 course topics rewritten as patient questions (`GALLM/data/p4_patient_questions.csv`, labelled as our rewording) |
| Cost per video | $0; only new scripts use free Groq tokens (about 11–22K per new question) |

Rejected, with reasons: generated AI video clips (cost, no control of objects or gestures, stray text); a 3D kit from
ready-made parts (no publishable adult characters with talking mouths or blinking; half the gestures still custom work);
a designer-built 2D rig (needs a designer and paid tools).

## 3. How one question becomes a video

```
question ─┬─ already in the library? ── yes → show the approved video
          └─ no → Part 1 facts → Part 2 checked script → director picks scenes
                   → voices → speech checks → timeline → actor kit, frame by frame → video
                   → shown with "not yet reviewed" → a person approves → added to the library
```

## 4. What stays and what changes

| Part | What happens |
|---|---|
| Graph, Part 1 (`graph_facts.py`) | Unchanged |
| Part 2 (`graph_script.py`) | Done: new closing line; empty-answer crash fixed. **Week 2:** check every line claim by claim, and each claim also against the other facts on the topic (section 8) |
| Director (`director.py`) | **This week:** a new score (section 8). Then GAAPO improves the director's instructions |
| Video (`make_video.py` + `video_kit/`) | The build below |

## 5. Build plan to Thursday

Test scripts: alcohol, exercise, ICD (`icd-driving-v5`). The ICD script has one wrong line (section 9), so it is **test
material only**.

| Day | Build | Done when |
|---|---|---|
| ✅ **Sat 21** | Actor kit drawn in code; style sheet approved | Done |
| **Sun 22** | Timeline (`timeline.json`); doctor voice + a second voice for the patient's question; slower speech (speed 0.8); checks that **stop** the video; a separate folder per run; the 6 `make_video.py` fixes. Director's new score, with tests | The ICD video plays (still as slides), every line passes; a failed check leaves **no** video |
| ✅ **Mon 23** (done Sat 21) | The kit moves: poses blend over time, blinking, breathing, mouths follow the voice's loudness; the kit replaces the slides; the storyboard (director's, or made by code) drives the scene | The ICD video shows the doctor and patient talking |
| **Tue 24** | **Meeting:** show the ICD video and the style sheet; confirm the style with the team. Graph check of the 44 questions (no AI tokens); pick the 3 question sets; start writing their scripts in the background | 3 sets fixed; scripts being written |
| **Wed 25** | Only approved text on screen; warned-against objects blocked; evidence record; captions file (.vtt) and transcript | Every check in section 6 runs |
| **Thu 26** | Run the 3 test scripts end to end | 3 videos, one command each, all checks passing |

**Honest limit:** writing scripts for the question sets takes about 3 days of Groq's free limit, so the **GAAPO run on the
director will most likely land early in week 2**, not by Thursday.

**Week 2:** GAAPO on the director (start vs winner) · Part 2 claim-by-claim check, then a proof run with pass marks set
before the run · writer score and GAAPO on the writer · live mode (library matching + "not yet reviewed" label) · review
page (approve / reject; shows each source's age warning) · library matching test on tricky question pairs · first
PEMAT-A/V scoring and viewer test · then Rhubarb mouth shapes, text reading (OCR) on frames, German.

## 6. What stops a video

Every check gives **PASS**, **CONTENT_FAIL** (wrong content) or **TOOL_ERROR** (the check could not run). Only all-PASS makes a video.

| # | Check | Fails when |
|---|---|---|
| 1 | Each spoken line (Whisper `small.en`) | Any word is added, dropped or changed |
| 2 | The whole soundtrack | The final mix does not say the script in order |
| 3 | Captions | Not exactly the script text (built by code, so they always are) |
| 4 | Text on screen | Any text that is not on the approved list |
| 5 | Drawings and actions | A value is not on the director's lists, or an object shows what the line warns against |
| 6 | Reading time | A caption is on screen too short to read |

## 7. Approved texts

| Where | Text | Status |
|---|---|---|
| Spoken closing line | Fits the question; code picks one by type: symptoms "If your symptoms get worse, contact your care team straight away." · medicines "Talk to your care team before you change how you take your medicines." · daily life "Your care team can tell you what is right for you." · other "Your care team can tell you what this means for you." | ✅ approved 21 Nov, in Part 2 (`closing_for`) |
| On-screen label | "Made with AI · not checked by a doctor" | to confirm |
| Spoken label (first scene) | "This video was made with AI and uses a computer voice." | to confirm |
| Live label | "Automatic answer · not yet reviewed" | to confirm |
| Source line (on screen) | Organisation or authors, title, **year**, page, e.g. "Source: Vijgen et al., ICD driving recommendations (EHRA), 2009, p. 8"; definitions: "MeSH (U.S. National Library of Medicine)". No "needs review" text on screen | ✅ decided (R1) |
| Age warning (review report only) | Guideline older than **3 years** → "check for a newer version"; older than **5 years** → strong warning. A date is a reason to check, not proof the advice is wrong | ✅ decided (R1) |

## 8. Prompt engineering

GAAPO (the recipe maker) writes new versions of two sets of instructions and keeps the best-scoring ones. Research
items P1–P7 decided how each is scored.

| Instructions for | Changes | Score |
|---|---|---|
| **Director** (this week's score, run early week 2) | What the video shows | Code: the object fits what is said, "none" is fine, a warned-against object gives 0, the doctor points at the object when it is mentioned; variety no longer counts (P1, Houts 2006). AI judge: compares in **both orders**, a win counts only if both answers agree (P2, Zheng 2023). Groq can force the allowed values (P7; one test call first) |
| **Writer** (week 2) | How good the script is | Each line split into single claims, each checked against its fact **and** the other facts on the topic (P6, Peters & Chin-Yee 2025, FActScore); plain words = share of uncommon words (P5, wordfreq, data frozen at 2021); no points for longer lines (P2) |

**Questions and fair results (P3, P4):** 44 questions in 3 sets of about 15 (learn / choose / final test), related topics
kept in the same set. GAAPO learns on set 1, picks the winner on set 2, and set 3 is used **once** at the end. Every test
runs twice, and a difference only counts if it is bigger than the noise. The old 3 / 3 / 2 split was too small: a
"winner" could be pure luck.

**Cost (estimate):** scripts for the question sets about 450K Groq tokens (about 3 days of the free limit); each GAAPO
run needs its own yes, with its token estimate first.

## 9. Risks, said plainly

- **The judge misses lines that say more than their fact.** ICD v5: "If your ICD is replaced, wait four weeks", while the
  same guideline says "1 week when only the ICD is replaced". Our judge **already has rules** against wider claims and
  still passed it, so the fix must change how lines are checked (section 8), not add more wording. Until then, no script
  is published.
- **No clinician.** Published content rests on automatic checks and one non-medical reviewer. The label must say this.
- **Old guidance.** ICD driving rules come from a 2009 European paper. A German 2023 DGK guideline exists, but Part 1 only
  uses English statements, so it can never reach a script.
- **The kit is code.** A new gesture or object later means new code, by Claude or a developer.
- **Live mode needs a server** and will hit Groq's free daily limits once public. Who hosts it is a team question.
- **Wrong library matches.** The best published systems still match wrongly 15–28% of the time, so matching is strict
  and a new video is made whenever unsure (section 2).
- **PEMAT-A/V measures design, not truth.** A high score does not mean the facts are right or that patients understood;
  hence the viewer test and our own checks.

Sources for R1–R3: [Shekelle et al. 2001](https://jamanetwork.com/journals/jama/fullarticle/194222) ·
[Vernooij et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC3904688/) ·
[AHRQ PEMAT-A/V](https://www.ahrq.gov/health-literacy/patient-education/pemat-av.html) ·
[ADHD TikTok PEMAT study](https://infodemiology.jmir.org/2025/1/e75973) ·
[Luo et al. 2015, SimQ](https://www.jmir.org/2015/2/e43/) ·
[McCreery et al., medical question similarity](https://arxiv.org/html/2008.13546)

## 10. Open before or during the build

- **Topics:** a graph check of the 44 questions (a script, no AI tokens; GraphDB must be open; needs your yes).
- **Downloads that need your yes:** Whisper `small.en` (about 480 MB, Sunday) and the `wordfreq` package (week 2).
- **Team, at Tuesday's meeting:** confirm the cartoon style, who hosts live mode, and who can be the second PEMAT rater.
- **Research:** none left before the build (P1–P7, A1–A6, B1–B4, R1–R3 done).
