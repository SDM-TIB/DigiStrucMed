# Fact-checked patient-education videos

From a patient question to a checked, animated patient video: guideline facts come from a knowledge graph, one model
(gpt-oss-20b) writes one plain line per fact, and a different model (gpt-oss-120b) checks every line against its own fact.
Code then picks what each scene shows from fixed lists and makes the video: two voices, a word-by-word speech check, and a
cartoon doctor and patient drawn in code. Test material only: no clinician has checked the scripts.
Code, how it works and how to run it: [GALLM/README.md](GALLM/README.md). Plan and decisions: [GALLM/VIDEO_PLAN.md](GALLM/VIDEO_PLAN.md).

`video/beta-blockers.mp4` is an example video made earlier from a checked script, before the animated version.
