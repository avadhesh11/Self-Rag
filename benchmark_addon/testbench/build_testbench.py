"""
Generate the PDF testbench: 4 reference PDFs about FICTIONAL entities plus
testbench/questions.jsonl.

Why fictional? A model cannot know the Duchy of Marrowmere or the ZX-9 battery,
so (a) a no-retrieval baseline should score near zero, (b) any correct answer
from a retrieval system is genuinely grounded in the PDF, and (c) "unanswerable"
questions really are unanswerable.

Question types (each mirrors a column of the paper's Table 2):
  short         single-fact lookup            (PopQA / TriviaQA style)
  multi         needs 2+ facts or arithmetic  (multi-chunk reasoning)
  tf            true/false claim              (PubHealth style)
  mc            4-option multiple choice      (ARC-Challenge style)
  unanswerable  answer is NOT in any PDF      (tests abstention / hallucination)
  general       common knowledge, no PDF need (tests adaptive retrieval)
  paper         facts from the Self-RAG paper PDF you already have (data/2310.11511v1.pdf)

Each question has `golds` (any substring => correct) and `evidence` (a phrase
that must appear in a retrieved passage => retrieval hit).

    python testbench/build_testbench.py
"""
import json
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

# Resolve output paths relative to this file, not the caller's cwd.
_HERE = Path(__file__).resolve().parent
OUT_PDF = _HERE.parent / "data/testbench"
OUT_Q = _HERE / "questions.jsonl"

DOCS = {
 "kestrel_valley_history.pdf": ("A Short History of Kestrel Valley", [
  "Kestrel Valley is a river town of 18,420 residents in the northern highlands of the fictional Duchy of Marrowmere. It was founded in 1847 by the surveyor Ilsabet Orne, who chose the site because two tributaries of the Quillon River meet there.",
  "The first public building was the grain exchange, completed in 1853 and designed by the architect Teodor Valk. The exchange stood until a fire destroyed its east wing in 1911, and the surviving hall now houses the town's Museum of Weights and Measures.",
  "The economy of Kestrel Valley depended on copper smelting from 1880 until the last furnace closed in 1964. The closure cost roughly 2,300 jobs. In response the council, led by Mayor Brannoch Pell, created the Kestrel Textile Cooperative in 1967, which today employs about 900 people.",
  "The Harrowgate Bridge, opened in 1902, spans 214 metres and was the longest suspension bridge in Marrowmere for 31 years. It was overtaken in 1933 by the Sorrel Span, which measures 268 metres.",
  "The town hosts the annual Lantern Fair every October. It began in 1921 as a harvest celebration organised by the schoolteacher Maud Ferreby and now draws around 40,000 visitors over four days.",
  "The current mayor, Odalys Winthrop, took office in 2019. Her administration opened the Quillon Riverwalk in 2022 and set a goal of planting 5,000 trees by 2030.",
  "Winters are cold: the average January temperature is minus 6 degrees Celsius, while July averages 19 degrees Celsius. Annual rainfall is 740 millimetres, and the wettest month is May.",
 ]),
 "aurora_zx9_technical_note.pdf": ("Aurora Cell Technologies: ZX9 Solid State Battery Technical Note", [
  "Aurora Cell Technologies, headquartered in Tallenport, released the ZX9 solid state battery in March 2025. The ZX9 stores 412 watt hours per kilogram, roughly 40 percent more than the company's previous ZX7 model, which stored 294 watt hours per kilogram.",
  "The electrolyte of the ZX9 is a sulfide glass named Kalmar3. Kalmar3 remains stable up to 210 degrees Celsius, and the cell is rated for 1,800 full charge cycles before capacity falls to 80 percent of its original value.",
  "In fast charging tests conducted at the Ostrava Test Facility, a ZX9 pack reached 80 percent charge in 11 minutes. Engineers noted that cell temperature peaked at 47 degrees Celsius during the test.",
  "Each ZX9 module weighs 2.6 kilograms and contains 24 cells. A standard vehicle pack uses 30 modules.",
  "Production began at the Tallenport plant, which has an annual capacity of 90,000 packs. A second plant in Nordhaven is scheduled to open in 2027 and will add 150,000 packs per year.",
  "Aurora's chief scientist, Dr. Priya Anselm, led the electrolyte research programme. The company holds 63 patents related to the ZX series.",
  "Known limitations: the ZX9 loses performance below minus 20 degrees Celsius, and the module cost is currently 38 percent higher than comparable lithium ion modules.",
 ]),
 "tarn2_lumivex_trial.pdf": ("The TARN2 Trial of Lumivex for Bone Density Loss", [
  "The TARN2 trial evaluated Lumivex, a vitamin D analogue, in 1,240 adults aged 50 to 75 with mild bone density loss. Participants were randomly assigned to Lumivex 20 micrograms daily or placebo for 18 months, and the trial was run at 14 clinics in the Halden region.",
  "The primary outcome was change in lumbar spine bone density. The Lumivex group gained 2.9 percent on average, compared with a gain of 0.4 percent in the placebo group. The difference was statistically significant.",
  "Fractures were a secondary outcome. There were 21 fractures among 620 Lumivex participants and 38 fractures among 620 placebo participants.",
  "Adverse events were mostly mild. Nausea occurred in 9 percent of Lumivex participants and 6 percent of placebo participants. In the Lumivex group, 11 participants developed raised blood calcium and left the trial, compared with 2 in the placebo group.",
  "The trial excluded people with kidney disease, and the authors caution that the results should not be applied to that group. The effect on hip bone density was not measured. Lumivex was not tested in children.",
  "The study was funded by the Halden Institute for Bone Health, and the authors reported no conflicts of interest. A follow up study, TARN3, will test a 40 microgram dose.",
 ]),
 "halcyon4_station_handbook.pdf": ("Halcyon4 Orbital Station: Crew Handbook", [
  "Halcyon4 is a research station orbiting at an altitude of 410 kilometres with an orbital period of 92 minutes. It carries a standing crew of six, rotated every 180 days. The station commander is Captain Ines Marchetti and her deputy is flight engineer Dov Haldane.",
  "Water is recycled on board: the Reclaimer unit recovers 94 percent of all water, including moisture from crew breath and urine. The remaining 6 percent is replaced by resupply flights, which arrive every 75 days.",
  "Exercise is mandatory. Each crew member must complete 2 hours of exercise per day, using the treadmill, the resistance device, or the cycle ergometer, to reduce bone and muscle loss in microgravity.",
  "In case of a hull pressure alarm, crew must first move to the Haven module, seal the hatch behind them, and only then report the alarm to ground control. The Haven module contains 72 hours of emergency oxygen.",
  "The solar arrays generate 84 kilowatts in sunlight. During the 36 minutes of each orbit spent in the shadow of Earth, power comes from lithium batteries.",
  "Crew eat three scheduled meals daily, and the galley stocks 210 menu items. Fresh lettuce is grown in the Verdant rack and harvested every 28 days.",
 ]),
}

ARC_INST = ("Given four answer candidates, A, B, C and D, choose the best answer choice. "
            "Please answer with the capitalized alphabet only, without adding any extra phrase or period.")
PUB_INST = ("Is the following statement correct or not? Say true if it's correct; otherwise, say false. "
            "Don't capitalize or add periods, just say \"true\" or \"false\".")

# (id, type, source_pdf, question, golds, evidence)
S = "short"; M = "multi"
Q = [
 # ---- Kestrel Valley ----
 ("kv01", S, "kestrel_valley_history.pdf", "Who founded the town of Kestrel Valley?", ["Ilsabet Orne", "Orne"], "founded in 1847 by the surveyor"),
 ("kv02", S, "kestrel_valley_history.pdf", "In what year did the last copper furnace in Kestrel Valley close?", ["1964"], "last furnace closed in 1964"),
 ("kv03", S, "kestrel_valley_history.pdf", "How long is the Harrowgate Bridge in metres?", ["214"], "spans 214 metres"),
 ("kv04", S, "kestrel_valley_history.pdf", "Which architect designed the Kestrel Valley grain exchange?", ["Teodor Valk", "Valk"], "designed by the architect Teodor Valk"),
 ("kv05", S, "kestrel_valley_history.pdf", "Which month is the wettest in Kestrel Valley?", ["May"], "wettest month is May"),
 ("kv06", S, "kestrel_valley_history.pdf", "Who organised the first Lantern Fair in Kestrel Valley?", ["Maud Ferreby", "Ferreby"], "organised by the schoolteacher Maud Ferreby"),
 ("kv07", M, "kestrel_valley_history.pdf", "How many years old was the Lantern Fair when the current mayor of Kestrel Valley took office?", ["98"], ["began in 1921", "took office in 2019"]),
 ("kv08", M, "kestrel_valley_history.pdf", "By how many metres is the Sorrel Span longer than the Harrowgate Bridge?", ["54"], ["spans 214 metres", "measures 268 metres"]),
 ("kv09", M, "kestrel_valley_history.pdf", "How many years after Kestrel Valley was founded did its last copper furnace close?", ["117"], ["founded in 1847", "last furnace closed in 1964"]),
 # ---- Aurora ZX9 ----
 ("az01", S, "aurora_zx9_technical_note.pdf", "How many watt hours per kilogram does the ZX9 battery store?", ["412"], "stores 412 watt hours per kilogram"),
 ("az02", S, "aurora_zx9_technical_note.pdf", "What is the name of the ZX9 electrolyte?", ["Kalmar"], "sulfide glass named Kalmar3"),
 ("az03", S, "aurora_zx9_technical_note.pdf", "Who led the electrolyte research programme at Aurora Cell Technologies?", ["Anselm"], "Priya Anselm"),
 ("az04", S, "aurora_zx9_technical_note.pdf", "Where were the ZX9 fast charging tests conducted?", ["Ostrava"], "Ostrava Test Facility"),
 ("az05", S, "aurora_zx9_technical_note.pdf", "How many patents does Aurora hold related to the ZX series?", ["63"], "holds 63 patents"),
 ("az06", M, "aurora_zx9_technical_note.pdf", "How many cells are in a standard ZX9 vehicle pack?", ["720"], ["contains 24 cells", "pack uses 30 modules"]),
 ("az07", M, "aurora_zx9_technical_note.pdf", "What will the combined annual capacity of Aurora's two plants be, in packs, once the second plant opens?", ["240,000", "240000", "240 000"], ["annual capacity of 90,000 packs", "add 150,000 packs per year"]),
 ("az08", M, "aurora_zx9_technical_note.pdf", "What is the total mass in kilograms of the modules in a standard ZX9 vehicle pack?", ["78"], ["weighs 2.6 kilograms", "pack uses 30 modules"]),
 # ---- TARN2 ----
 ("tr01", S, "tarn2_lumivex_trial.pdf", "How many adults took part in the TARN2 trial?", ["1,240", "1240"], "Lumivex, a vitamin D analogue, in 1,240 adults"),
 ("tr02", S, "tarn2_lumivex_trial.pdf", "How many months did the TARN2 trial last?", ["18"], "placebo for 18 months"),
 ("tr03", S, "tarn2_lumivex_trial.pdf", "What daily dose of Lumivex was given in the TARN2 trial, in micrograms?", ["20"], "Lumivex 20 micrograms daily"),
 ("tr04", M, "tarn2_lumivex_trial.pdf", "How many fractures occurred in total across both groups of the TARN2 trial?", ["59"], ["21 fractures among 620 Lumivex", "38 fractures among 620 placebo"]),
 ("tr05", M, "tarn2_lumivex_trial.pdf", "How many more participants in the Lumivex group than in the placebo group left the trial because of raised blood calcium?", ["9", "nine"], ["11 participants developed raised blood calcium", "compared with 2 in the placebo group"]),
 # ---- Halcyon4 ----
 ("hc01", S, "halcyon4_station_handbook.pdf", "At what altitude in kilometres does Halcyon4 orbit?", ["410"], "altitude of 410 kilometres"),
 ("hc02", S, "halcyon4_station_handbook.pdf", "Who is the commander of the Halcyon4 station?", ["Marchetti"], "Captain Ines Marchetti"),
 ("hc03", S, "halcyon4_station_handbook.pdf", "How often do resupply flights arrive at Halcyon4, in days?", ["75"], "arrive every 75 days"),
 ("hc04", S, "halcyon4_station_handbook.pdf", "How many crew members form the standing crew of Halcyon4?", ["six", "6"], "standing crew of six"),
 ("hc05", M, "halcyon4_station_handbook.pdf", "For how many minutes of each orbit is Halcyon4 in sunlight?", ["56"], ["orbital period of 92 minutes", "36 minutes of each orbit"]),
]

TF = [  # (id, source, claim, gold, evidence)
 ("tf01", "tarn2_lumivex_trial.pdf", "In the TARN2 trial, Lumivex participants had fewer fractures than placebo participants.", "true", "21 fractures among 620 Lumivex participants"),
 ("tf02", "tarn2_lumivex_trial.pdf", "The TARN2 trial measured the effect of Lumivex on hip bone density.", "false", "effect on hip bone density was not measured"),
 ("tf03", "tarn2_lumivex_trial.pdf", "Nausea was more common in the Lumivex group than in the placebo group in the TARN2 trial.", "true", "Nausea occurred in 9 percent"),
 ("tf04", "tarn2_lumivex_trial.pdf", "Children were included as participants in the TARN2 trial.", "false", "Lumivex was not tested in children"),
 ("tf05", "tarn2_lumivex_trial.pdf", "The TARN2 trial found no significant difference in lumbar spine bone density between the two groups.", "false", "difference was statistically significant"),
 ("tf06", "aurora_zx9_technical_note.pdf", "The ZX9 battery stores more energy per kilogram than the ZX7.", "true", "roughly 40 percent more than the company's previous ZX7"),
 ("tf07", "halcyon4_station_handbook.pdf", "Crew on Halcyon4 are allowed to skip exercise on days when they are tired.", "false", "must complete 2 hours of exercise per day"),
]
MC = [  # (id, source, question, options A-D, gold, evidence)
 ("mc01", "halcyon4_station_handbook.pdf", "According to the Halcyon4 handbook, what must the crew do first when a hull pressure alarm sounds?", ["Report the alarm to ground control", "Move to the Haven module", "Begin emergency exercise", "Switch to battery power"], "B", "must first move to the Haven module"),
 ("mc02", "halcyon4_station_handbook.pdf", "What percentage of water does the Halcyon4 Reclaimer unit recover?", ["84 percent", "92 percent", "94 percent", "99 percent"], "C", "recovers 94 percent of all water"),
 ("mc03", "halcyon4_station_handbook.pdf", "How often is lettuce harvested from the Verdant rack on Halcyon4?", ["Every 14 days", "Every 28 days", "Every 36 days", "Every 75 days"], "B", "harvested every 28 days"),
 ("mc04", "halcyon4_station_handbook.pdf", "How many hours of emergency oxygen does the Haven module contain?", ["36", "48", "72", "94"], "C", "contains 72 hours of emergency oxygen"),
 ("mc05", "halcyon4_station_handbook.pdf", "Where does Halcyon4 get power while it is in the shadow of Earth?", ["Solar arrays", "Lithium batteries", "The Reclaimer unit", "The Verdant rack"], "B", "power comes from lithium batteries"),
]
UNANS = [  # answer is NOT in any PDF
 ("un01", "kestrel_valley_history.pdf", "Who designed the Harrowgate Bridge?"),
 ("un02", "aurora_zx9_technical_note.pdf", "What is the price of a ZX9 module in US dollars?"),
 ("un03", "tarn2_lumivex_trial.pdf", "How many participants in the TARN2 trial were older than 70?"),
 ("un04", "halcyon4_station_handbook.pdf", "Who is the chef responsible for the Halcyon4 galley?"),
 ("un05", "kestrel_valley_history.pdf", "What was the population of Kestrel Valley in 1900?"),
 ("un06", "aurora_zx9_technical_note.pdf", "In what year was Aurora Cell Technologies founded?"),
]
GENERAL = [
 ("ge01", "What is the capital of Australia?", ["Canberra"]),
 ("ge02", "What is 17 multiplied by 6?", ["102"]),
 ("ge03", "Who wrote the play Hamlet?", ["Shakespeare"]),
 ("ge04", "What is the chemical symbol for gold?", ["Au"]),
]
PAPER = [  # facts verified against the Self-RAG paper text (2310.11511)
 ("pp01", "How many rare entity queries are in the PopQA long tail subset used to evaluate Self-RAG?", ["1,399", "1399"], "1,399"),
 ("pp02", "How many TriviaQA test queries were used to evaluate Self-RAG?", ["11,313", "11313"], "11,313"),
 ("pp03", "How many instruction output pairs make up the Self-RAG training data?", ["150k", "150,000"], "150k"),
 ("pp04", "Which retriever does Self-RAG use by default?", ["Contriever"], "Contriever"),
 ("pp05", "How many top documents does Self-RAG retrieve by default using its retriever?", ["five", "5"], "top five"),
 ("pp06", "What maximum number of new tokens was set for PubHealth and ARC-Challenge for all baselines?", ["50"], "maximum new token length to 50"),
]


def build_pdfs():
    OUT_PDF.mkdir(parents=True, exist_ok=True)
    ss = getSampleStyleSheet()
    for fname, (title, paras) in DOCS.items():
        doc = SimpleDocTemplate(str(OUT_PDF / fname), pagesize=A4, invariant=1,
                                title=title, author="testbench")
        flow = [Paragraph(title, ss["Title"]), Spacer(1, 12)]
        for p in paras:
            flow += [Paragraph(p, ss["BodyText"]), Spacer(1, 8)]
        doc.build(flow)


def build_questions() -> list[dict]:
    rows = []
    for qid, typ, src, q, golds, ev in Q:
        rows.append(dict(id=qid, type=typ, source=src, question=q, golds=golds, evidence=ev if isinstance(ev, list) else [ev]))
    for qid, src, claim, gold, ev in TF:
        rows.append(dict(id=qid, type="tf", source=src, question=claim, instruction=f"{PUB_INST}\n\n### Input:\n{claim}", golds=[gold], evidence=[ev]))
    for qid, src, q, opts, gold, ev in MC:
        o = "".join(f"\n{l}: {t}" for l, t in zip("ABCD", opts))
        rows.append(dict(id=qid, type="mc", source=src, question=q, instruction=f"{ARC_INST}\n\n### Input:\n{q}{o}", golds=[gold], evidence=[ev]))
    for qid, src, q in UNANS:
        rows.append(dict(id=qid, type="unanswerable", source=src, question=q, golds=[], evidence=[]))
    for qid, q, golds in GENERAL:
        rows.append(dict(id=qid, type="general", source=None, question=q, golds=golds, evidence=[]))
    for qid, q, golds, ev in PAPER:
        rows.append(dict(id=qid, type="paper", source="2310.11511v1.pdf", question=q, golds=golds, evidence=[ev]))
    return rows


if __name__ == "__main__":
    build_pdfs()
    rows = build_questions()
    OUT_Q.parent.mkdir(parents=True, exist_ok=True)
    OUT_Q.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    from collections import Counter
    print(f"wrote {len(DOCS)} PDFs to {OUT_PDF}/ and {len(rows)} questions to {OUT_Q}")
    print(dict(Counter(r["type"] for r in rows)))
