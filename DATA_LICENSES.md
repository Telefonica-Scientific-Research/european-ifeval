# Dataset licensing and attribution

The files under `data/` are normalized from the upstream benchmark datasets
listed below. They are not covered by the repository's Apache 2.0 code
license. Each dataset retains its upstream license and attribution requirements.

European IFEval changes the record format and harmonizes instruction-checker
identifiers, arguments and response-language constraints. The per-language
source and other provenance details are recorded in `data/source.json`.

| European IFEval files | Upstream dataset | Publisher or creator | Upstream license |
|---|---|---|---|
| `ca.jsonl` | [IFEval_ca](https://huggingface.co/datasets/projecte-aina/IFEval_ca) | Language Technologies Unit, BSC-CNS | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `es.jsonl` | [IFEval_es](https://huggingface.co/datasets/BSC-LT/IFEval_es) | Language Technologies Unit, BSC-CNS | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `el.jsonl` | [IFEval Greek](https://huggingface.co/datasets/ilsp/ifeval_greek) | ILSP/Athena Research Center | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| `de.jsonl` | [de_ifeval](https://huggingface.co/datasets/jzhang86/de_ifeval) | jzhang86 | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| `da.jsonl` | [ifeval-da](https://huggingface.co/datasets/danish-foundation-models/ifeval-da) | Danish Foundation Models | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| `et.jsonl` | [ifeval_et](https://huggingface.co/datasets/tartuNLP/ifeval_et) | TartuNLP | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| `fi.jsonl`, `sv.jsonl` | [ifeval_mt](https://huggingface.co/datasets/LumiOpen/ifeval_mt) | LumiOpen | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| `cs.jsonl`, `fr.jsonl`, `hu.jsonl`, `it.jsonl`, `nl.jsonl`, `pl.jsonl`, `pt.jsonl`, `ro.jsonl`, `ru.jsonl`, `uk.jsonl` | [Marco-Bench-MIF](https://huggingface.co/datasets/AIDC-AI/Marco-Bench-MIF) | AIDC-AI | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| `ast.jsonl`, `bg.jsonl`, `bs.jsonl`, `cy.jsonl`, `fo.jsonl`, `fur.jsonl`, `ga.jsonl`, `gd.jsonl`, `hr.jsonl`, `is.jsonl`, `ka.jsonl`, `lb.jsonl`, `lij.jsonl`, `lmo.jsonl`, `lt.jsonl`, `ltg.jsonl`, `lv.jsonl`, `mk.jsonl`, `mt.jsonl`, `nn.jsonl`, `no.jsonl`, `oc.jsonl`, `sc.jsonl`, `scn.jsonl`, `sk.jsonl`, `sl.jsonl`, `sq.jsonl`, `sr.jsonl`, `szl.jsonl`, `vec.jsonl` | [MultiIFEval](https://huggingface.co/datasets/danish-foundation-models/multi-ifeval) | Dan Saattrup Smart, Alexandra Institute | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| `eu.jsonl`, `gl.jsonl` | [IFEval EU](https://huggingface.co/datasets/HiTZ/ifeval_eu), [IFEval GL](https://huggingface.co/datasets/HiTZ/ifeval_gl) | HiTZ Center | No explicit license stated by the upstream repositories |

CC BY-NC-SA 4.0 permits sharing and adaptation only for non-commercial
purposes, requires attribution and requires adaptations to use the same
license. CC BY 4.0 requires attribution and an indication of modifications.

We are awaiting confirmation from HiTZ regarding the licenses for the Basque
and Galician datasets.
