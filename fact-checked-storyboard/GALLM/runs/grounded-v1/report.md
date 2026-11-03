# Grounded storyboard prompts: first run (6 medicine topics)

Pipeline: request → guideline facts from the knowledge graph → editor (gpt-oss-120b, Mahsa's best GAAPO prompt) → every sentence fact-checked → unsupported sentences rewritten or removed.

## ruler-012: Create a brief patient-education scene that explains how ARNIs (for example sacubitril/valsartan) help the heart.

**Title:** How ARNIs Help Your Heart  
**Narration:** An ARNI mixes a blood‑vessel relaxer with a salt‑removing drug. It widens arteries and removes fluid, making the heart pump easier.  
**Visual:** Illustration of heart, widened arteries, and fluid leaving body.  
**On screen:** ARNI helps heart pump

| Sentence | Check | Guideline fact | Grade |
|---|---|---|---|
| An ARNI mixes a blood‑vessel relaxer with a salt‑removing drug. | SUPPORTED | ARNIs sacubitril combined with valsartan Entresto Combines an ARB with a drug that helps the arteries widen and rid the body of salt. Make it easier for the hea | patient material |
| It widens arteries and removes fluid, making the heart pump easier. | SUPPORTED | ARNIs sacubitril combined with valsartan Entresto Combines an ARB with a drug that helps the arteries widen and rid the body of salt. Make it easier for the hea | patient material |
| ARNI helps heart pump | SUPPORTED | ARNIs sacubitril combined with valsartan Entresto Combines an ARB with a drug that helps the arteries widen and rid the body of salt. Make it easier for the hea | patient material |

Flagged in the first draft and fixed: none

**Fine-tuned model, same request (before):** {"title":"ARNI Help Heart Function", "-scenes": { " , "narrATION":"The medication helps to slow down the rate of ventricular dilation by reducing wall stress on the left side.", "visual_description":"Helps reduce strain on the mitral valve", "on_screen_TEXT":" \ud83e\ude00d7a5fde7789efacdb825ae7ecf43b8c0f82206bd8674eb249993ea45592298fdadac Based on the information above, can you please provide mor

## ruler-013: Create a brief patient-education scene that explains in simple words how ACE inhibitors work.

**Title:** How ACE Inhibitors Help Your Heart  
**Narration:** ACE inhibitors relax your blood vessels, lower blood pressure, and make it easier for your heart to pump blood.  
**Visual:** Illustration of relaxed blood vessels and a heart pumping smoothly.  
**On screen:** ACE inhibitors help the heart

| Sentence | Check | Guideline fact | Grade |
|---|---|---|---|
| ACE inhibitors relax your blood vessels, lower blood pressure, and make it easier for your heart to pump blood. | SUPPORTED | ACE inhibitors enalapril, captopril, lisinopril Vasotec, Capoten, Zestril Relax blood vessels. Lower blood pressure. Make it easier for the heart to pump blood. | patient material |
| ACE inhibitors help the heart | SUPPORTED | ACE inhibitors enalapril, captopril, lisinopril Vasotec, Capoten, Zestril Relax blood vessels. Lower blood pressure. Make it easier for the heart to pump blood. | patient material |

Flagged in the first draft and fixed: none

**Fine-tuned model, same request (before):** {"title":"ACE Inhibitors Explained", "-scenes": { " , "narrATION":"The goal of an ACE inhibitor is to block the action of angiotensin-converting enzyme.", "visual_description":"A medication called an ACE inhibitor blocks the conversion of angietans from deoxotension to angioedema.", on_screen_text":"","on_screen_color":0xFF000000} {"title":"How it works:", "scenes": { " , "n arrivation":" Medicati

## ruler-014: Create a brief patient-education scene that explains how beta-blockers protect the heart in heart failure.

**Title:** How Beta‑Blockers Help the Heart  
**Narration:** Beta‑blockers slow how fast your heart beats and lower blood pressure. This makes it easier for the heart to pump blood and can even help repair some damage.  
**Visual:** Illustration of a heart with a calm pulse and lower pressure gauge.  
**On screen:** Beta‑blockers protect the heart

| Sentence | Check | Guideline fact | Grade |
|---|---|---|---|
| Beta‑blockers slow how fast your heart beats and lower blood pressure. | SUPPORTED | Beta blocker Beta blockers slow the heart rate, lower blood pressure and may reverse some of the heart damage. bisoprolol (Zebeta) carvedilol (Coreg) metoprolol | patient material |
| This makes it easier for the heart to pump blood and can even help repair some damage. | SUPPORTED | Beta blocker Beta blockers slow the heart rate, lower blood pressure and may reverse some of the heart damage. bisoprolol (Zebeta) carvedilol (Coreg) metoprolol | patient material |
| Beta‑blockers protect the heart | SUPPORTED | Beta blocker Beta blockers slow the heart rate, lower blood pressure and may reverse some of the heart damage. bisoprolol (Zebeta) carvedilol (Coreg) metoprolol | patient material |

Flagged in the first draft and fixed: none

**Fine-tuned model, same request (before):** {"title":"Beta Blocker Therapy for Heart Failure", "-scenes": { " , "narrATION":"The goal of Beta blocker therapy in patients with HF is to reduce morbidity by reducing mortality.", "visual_description":"A medication called an ACE inhibitor helps relax blood vessels so they don't constrict too much when your body needs them more than usual.", "on_screen_TEXT":" \ud83e\ude0d Based on clinical exper

## ruler-015: Create a brief patient-education scene that explains how MRAs help the body get rid of extra fluid.

**Title:** How MRAs Remove Extra Fluid  
**Narration:** Mineralocorticoid antagonists help your body remove extra salt and fluid, so the heart can pump easier.  
**Visual:** Doctor showing a heart and water droplets disappearing  
**On screen:** MRA removes extra fluid

| Sentence | Check | Guideline fact | Grade |
|---|---|---|---|
| Mineralocorticoid antagonists help your body remove extra salt and fluid, so the heart can pump easier. | SUPPORTED | Mineralocorticoid Antagonists (MRA) spironolactone, eplerenone Aldactone, Inspra Help the body get rid of salt and fluids. Make it easier for the heart to pump  | patient material |
| MRA removes extra fluid | SUPPORTED | Mineralocorticoid Antagonists (MRA) spironolactone, eplerenone Aldactone, Inspra Help the body get rid of salt and fluids. Make it easier for the heart to pump  | patient material |

Flagged in the first draft and fixed: none

**Fine-tuned model, same request (before):** {"title":"Managing Excess Fluid", "-scenes": { " , "narrATION":I CD has been prescribed to reduce excess fluids from your diet.", "visual_description":The doctor will give you an IV line under the skin where they'll slowly remove some of these excess fluids over time.', "on_screen_TEXT":"","} {"title":"Removing Extra Fluids", ".scenes": { " , "n arrivation":E C G has helped many patients manage th

## ruler-016: Create a brief patient-education scene that explains SGLT2 inhibitors, which were first developed for diabetes, and how they help in heart failure.

**Title:** SGLT2 Inhibitors: From Diabetes to Heart Failure  
**Narration:** SGLT2 inhibitors were created to control blood sugar in diabetes. Studies show they also cut heart‑failure deaths and hospital stays, even for people without diabetes.  
**Visual:** Doctor holds a pill bottle next to a heart graphic.  
**On screen:** Helps heart failure

| Sentence | Check | Guideline fact | Grade |
|---|---|---|---|
| SGLT2 inhibitors were created to control blood sugar in diabetes. | SUPPORTED | Sodium-glucose cotransporter-2 (SGLT2) Inhibitor dapagliflozin, empagliflozin Farxiga, Jardiance Used to treat diabetes but are also to treat heart failure in p | patient material |
| Studies show they also cut heart‑failure deaths and hospital stays, even for people without diabetes. | SUPPORTED | Sodium-glucose cotransporter-2 (SGLT2) Inhibitor dapagliflozin, empagliflozin Farxiga, Jardiance Used to treat diabetes but are also to treat heart failure in p | patient material |
| Helps heart failure | SUPPORTED | Sodium-glucose cotransporter 2 (SGLT2) inhibitor Originally used to treat diabetes, these medicines have clear benefits for people with heart failure, including | patient material |

Flagged in the first draft and fixed: none

**Fine-tuned model, same request (before):** {"title":SGLT Inhibitors Explained", "-scenes": { " , "narrATION":Medication class IIa, patients should discuss their individual situation with an endocrinologist to determine appropriate therapy based on current clinical judgment.", "visual_description":Patients who have had adverse effects from ACE inhibitor therapy may benefit from alternative options such as ARBs or SGLTs like empagliflozineme

## ruler-017: Create a brief patient-education scene that explains why a high potassium level in the blood is dangerous and why it is checked.

**Title:** Understanding High Potassium  
**Narration:** High potassium can cause dangerous heart rhythm problems. Doctors check potassium levels to keep them normal and guide treatment.  
**Visual:** Doctor showing blood test results to patient.  
**On screen:** High Potassium Risks

| Sentence | Check | Guideline fact | Grade |
|---|---|---|---|
| High potassium can cause dangerous heart rhythm problems. | SUPPORTED | Acute hyperkalaemia: if ECG abnormalities present, then prevent arrhythmias by IV calcium. Intermediate strategies include: insulin/albuterol/sodium \text{HCO}_ | patient material |
| Doctors check potassium levels to keep them normal and guide treatment. | SUPPORTED | Careful observation for development of a variety of side effects, including renal dysfunction, electrolyte abnormalities and symptomatic hypotension, is recomme | patient material |
| High Potassium Risks | SUPPORTED | Acute hyperkalaemia: if ECG abnormalities present, then prevent arrhythmias by IV calcium. Intermediate strategies include: insulin/albuterol/sodium \text{HCO}_ | patient material |

Flagged in the first draft and fixed: none

**Fine-tuned model, same request (before):** {"title":"High Potassium Level", "-scenes": { " , "narrative":"A low to moderate hypokalemia can lead to serious complications such as muscle weakness, fatigue, dizziness, confusion, chest pain, palpitations, shortness of breath, nausea, vomiting, abdominal cramps, constipation, irregular heartbeats, QT prolongation on an electrocardiogram, and even cardiac arrest.", "visual_description":"Low pota
