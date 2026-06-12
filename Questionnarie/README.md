
# Questionnaire Knowledge Graph

This project builds and explores a questionnaire knowledge graph for DigiLearnHF evaluation data. It converts questionnaire response exports and a codebook into normalized CSV tables, maps those tables to RDF with RML/R2RML mappings, and supports exploration through GraphDB and a Jupyter notebook.

## Project Structure

```text
.
|-- export_dbml_tables.py                 # Builds normalized CSV tables from the evaluation export and codebook
|-- config.ini                            # RDFizer configuration
|-- TableVisualization-mappings.ttl       # RML/R2RML mappings from CSV tables to RDF
|-- QuestionariesOntology.ttl             # Questionnaire ontology
|-- QuestionaariesRule.nt                 # Additional RDF rules/data
|-- KG-QuestionnarieExploration.ipynb     # SPARQL and graph exploration notebook
|-- input-tables/                         # Generated CSV tables used by RDFizer
`-- rdf-dump/                             # RDFizer output
```

## Data Model

The knowledge graph represents:

- `Platform`: the platform used for questionnaire delivery.
- `User`: respondents.
- `Questionnaire`: questionnaire instruments such as SUS, uMARS, and Module Evaluation.
- `Question`: individual questionnaire questions.
- `Answer`: possible answer options.
- `Answered`: response events connecting users, questions, selected answers, platform, and questionnaire context.

The main namespace is:

```text
http://digistrucmed.org/questionnaire#
```

## Requirements

Install Python 3.10+ and the required Python packages:

```powershell
python -m pip install pandas openpyxl rdfizer SPARQLWrapper networkx matplotlib
```

GraphDB is optional for local RDF exploration, but required if you want to use the notebook SPARQL endpoint.

## Generate Input Tables

The source files expected by `export_dbml_tables.py` are:

```text
DigiLearnHF_evaluation_sosci.xlsx
codebook_DigiLearnHF_2025-09-11_12-21.csv
```

Run:

```powershell
python .\export_dbml_tables.py
```

This generates the RDFizer input tables in `input-tables/`:

```text
Platform.csv
User.csv
Questionnarie.csv
Questions.csv
Answer.csv
Answered.csv
```

Current generated row counts:

```text
Platform.csv: 1
User.csv: 273
Questionnarie.csv: 3
Questions.csv: 42
Answer.csv: 311
Answered.csv: 3167
```

## Build RDF

The RDFizer configuration is in `config.ini`. It writes RDF output to `rdf-dump/` and uses:

```text
TableVisualization-mappings.ttl
```

Run:

```powershell
python -m rdfizer -c .\config.ini
```

The main RDF dump is expected at:

```text
rdf-dump/KG-Questionnaire.nt
```

## Load Into GraphDB

Create or use a GraphDB repository and import:

```text
rdf-dump/KG-Questionnaire.nt
QuestionariesOntology.ttl
```

In the current notebook, the configured local repository endpoint is:

```text
http://localhost:7200/repositories/KG-Questionnarie
```

Note the spelling: the live GraphDB repository may use `KG-Questionnarie`, while the generated RDF dump is named `KG-Questionnaire.nt`.

## Explore the Knowledge Graph

Open:

```text
KG-QuestionnarieExploration.ipynb
```

The notebook uses `SPARQLWrapper`, `pandas`, `networkx`, and `matplotlib` to query the graph and compute network metrics such as degree centrality.

Example SPARQL endpoint setup:

```python
ENDPOINT_URL = "http://localhost:7200/repositories/KG-Questionnarie"

PREFIXES = """\
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX ds:   <http://digistrucmed.org/questionnaire#>
"""
```

## Core Relations

Important ontology and mapping predicates include:

```text
ds:belongsTo
ds:hasPossibleAnswer
ds:isAnswerTo
ds:answeredByUser
ds:answeredOnPlatform
ds:answeredInQuestionnaire
ds:hasQuestion
ds:hasAnswer
ds:rawResponseValue
ds:startTime
ds:endTime
ds:speed
```

## Typical Workflow

1. Place/update the evaluation workbook and codebook CSV in the project root.
2. Run `export_dbml_tables.py` to regenerate `input-tables/`.
3. Run RDFizer with `config.ini`.
4. Import the generated RDF into GraphDB.
5. Use `KG-QuestionnarieExploration.ipynb` for SPARQL queries and graph analysis.

## Notes

- The project intentionally keeps the spelling `Questionnarie` in several filenames and CSV columns because the existing mapping file expects those names.
- The codebook CSV is semicolon-separated.
- `input-tables/` is the source folder used by `TableVisualization-mappings.ttl`.
