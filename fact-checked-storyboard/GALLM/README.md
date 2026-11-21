# From a patient question to a checked patient video

A patient asks a question; the pipeline answers with a short animated video (a cartoon doctor explains to a patient) in which
every spoken line comes from a clinical-guideline fact in the knowledge graph and has been checked against that fact.

**Status (21 Nov 2026): test material only.** No clinician has checked these scripts. A known gap: the per-line judge can pass a
line that says more than its fact (ICD example: "If your ICD is replaced, wait four weeks", while the guideline says 1 week
when only the ICD is replaced). A claim-by-claim check is planned (`VIDEO_PLAN.md`, section 8); until then no video is published.

## Part 1 — facts from the knowledge graph (`graph_facts.py`, code only)

1. **Find concepts**: match the question's words to the graph's own names (spelling and word forms, abbreviations only in
   capitals, near-identical meaning match with a local MiniLM model). `find_concepts`
2. **Query the graph** with fixed SPARQL: Query A — claims (`CLAIMS`), Query B — recommendation readings (`READINGS`), plus
   `STATEMENTS` (sentence, grade, page) and `BROADER` (stop rule: step to a broader concept, at most 3 levels).
3. **Rank and filter**: direct before broader, specific concepts first, grade, recommendation before explanation; drop
   off-topic facts and near-copies; keep at most 10. `choose`
4. **Fact text with source**: triple, type, grade, guideline, page and the guideline sentence. `fact_text`

## Part 2 — from facts to a checked script (`graph_script.py`)

5. **Slots** (code): one per fact, with type, grade, caution ("may") and abbreviation meanings from the graph. `slots_of`
6. **Writer** gpt-oss-20b (Groq): one plain line per fact, at most 6 lines; skips off-topic, repeated or conflicting facts.
7. **Code checks**: numbers in the fact, caution kept, unclear abbreviations kept, on-screen words from line or fact.
8. **Judge** gpt-oss-120b (Groq, `new_judge.py`): each line against its own fact only; quote must be in the fact; asked three
   ways. A failed line is rewritten once with the reason, else dropped.
9. **Opening and closing** are fixed sentences written by code (no claim). The closing fits the question: code picks one by
   question type (symptoms, medicines, daily life, other). `closing_for`
10. **Video prompt**: built by code from the kept lines, with a source table.

`text_tools.py` holds small shared helpers (JSON reading, word normalisation, the MiniLM encoder).

## Part 3 — what the video shows (`director.py`)

11. **Storyboard**: for each line, a camera shot, what the doctor does, what the patient does and at most one object, chosen
    only from fixed lists (`SHOTS`, `DOCTOR`, `PATIENT`, `PROPS`). The director never writes words. `gate` rejects values not
    on the lists and objects a line warns against (no cigarette pack on "stop smoking"). `code_storyboard` makes one by code
    (no model): the object must fit the words of its line (`PROP_FITS`), and the doctor shows it when it is mentioned.
12. **Score for prompt optimisation** (`evolve_director.py`, GAAPO): code part 40% (objects fit, doctor shows the object,
    patient takes part) and an AI judge 60%, asked in both orders; a win counts only if both answers agree.

## The video (`make_video.py`, no model, no tokens)

Uses the open-source tools of [Lemo-Opuscar](https://github.com/lemomo-ai/lemo-opuscar) (Kokoro voices, Whisper, a
frame-by-frame HTML renderer, ffmpeg). Every check gives PASS, CONTENT_FAIL or TOOL_ERROR; only all-PASS makes a video,
otherwise the attempt is kept in `failed-<time>/` with a report.

1. **Voices**: the patient asks their own question (second voice), then the doctor says the checked lines; speed 0.8.
2. **Speech check** (`speech_check.py`, Whisper `small.en`): every line must say its script words exactly (numbers may be
   written as digits or words).
3. **Timeline** (`timeline.py`): one `timeline.json` from lines + storyboard + voice lengths; never shorter than the reading
   time of the text on screen.
4. **Speech check of the finished sound track**, piece by piece along the timeline (order, overlaps, cut-offs).
5. **Picture**: `video_kit/kit.js` draws the approved doctor, patient, room and 18 objects (no letters or logos) in code;
   `video_kit/scene.js` animates them from the timeline (poses blend, breathing, blinking, the speaker's mouth follows the
   loudness of their voice). Captions are the spoken words; a readable source line with the guideline's year; a
   reading-time check.
6. **Picture + sound** joined at -14 LUFS. Output in `runs/videos/<name>/`: `video.mp4`, `subtitles.srt`, `timeline.json`,
   `report.json` (with an age warning per source, for the reviewer only).

`video_kit/style_sheet.png` shows the approved look. The full plan, decisions and risks: `VIDEO_PLAN.md`.

## Run

Needs GraphDB with the repository `Guideline_KG` at `http://127.0.0.1:7200`, `GROQ_API_KEY` (environment or `GAAPO/.env`),
and Python with `openai`, `numpy`, `torch`, `transformers` (MiniLM in the local cache), `python-dotenv`, optionally `truststore`.
The video needs Lemo-Opuscar in `~/lemo-opuscar` (or `LEMO_OPUSCAR_HOME`) with its own Python, Node, ffmpeg and Whisper
`small.en`. Run from the folder that contains `GALLM/`:

    python -m GALLM.graph_script "When can I drive again after getting an ICD?" --name icd-driving
    <lemo-opuscar python> GALLM/make_video.py GALLM/runs/part2/icd-driving.json
    python GALLM/demo_page.py        # then open http://127.0.0.1:8770

The demo page shows every step live: the graph facts, every line with its checks, the storyboard, and a "Make the video"
button. Every model call is saved, so a question asked again costs no tokens.

## Tests

    python -m unittest GALLM.tests.test_graph_facts GALLM.tests.test_graph_script GALLM.tests.test_director GALLM.tests.test_timeline GALLM.tests.test_demo_page

`evolve_director.py` and 4 tests in `tests/test_director.py` need Mahsa's GAAPO optimiser: the folder `GAAPO/` from
`SDM-TIB/MahsaThesis-DigiStrucMed`, placed next to `GALLM/`. Without it those 4 tests stop with "No module named 'GAAPO'";
the other 89 pass.

`tests/test_new_judge.py` is written for pytest. 2 of its 10 tests still import the older pipeline (`grounded_pipeline`),
which is no longer in this folder, so they fail; the other 8 pass.

## Not in this repository

The graph's name index (`data/graph_names*.json/.npy`, built from GraphDB on the first run; it contains UMLS names, which stay
local), run outputs and keys.
