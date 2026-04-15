"""
Thesis pipeline package. All implementation code lives in subpackages only:

- ``pipeline.step1`` — PDF text + table extraction
- ``pipeline.step2`` — normalization + extraction plan + ``guideline_config.json``
- ``pipeline.step3`` — UMLS linking + optional Llama 3.1 disambiguation

There are no other Python modules beside this file at the ``pipeline/`` root.
"""
