# Branch `fact-checked-storyboard`

This branch adds **fact-checked storyboards for heart-failure patient videos**.

Each storyboard is written from guideline facts in the knowledge graph (GuideLineKG). Every sentence is checked
against those facts, and what is not supported is repaired or removed. GAAPO tunes the writer's prompt.

Everything is in [`fact-checked-storyboard/`](fact-checked-storyboard/README.md): the pipeline, the results, the
limits and how to run it. The rest of the repository is unchanged from `main`.
