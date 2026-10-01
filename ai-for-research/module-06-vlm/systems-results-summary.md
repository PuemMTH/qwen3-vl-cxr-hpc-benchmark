# Systems Results Summary — Pilot Matrix 2026-09-23

**Study status:** `development_pilot_only` — ยังไม่เปิด confirmatory test set
(`confirmatory_test_accessed: false`)

**Aggregate artifact:**
`vlm-cxr-benchmark/runs/pilot-matrix-20260923/systems-summary.json` (+ `.csv`)
SHA-256 `525f74bf…a7c95` / `85a0dcf0…5b048` (บันทึกใน `design-decision-log.md`)

ทุกตารางในเอกสารนี้มีบล็อก **ที่มา** กำกับ ระบุ code ที่คำนวณค่าและ reference
environment ที่วัดจริง

---

## 0. Provenance conventions

### 0.1 Reference environments

ทั้งสองชุดบันทึกอัตโนมัติในฟิลด์ `.environment` ของทุก
`round-NN/intermediate/summary.json`

**`ENV-LOCAL`** — เครื่องส่วนตัว
| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070, 12,343,050,240 B, CC 12.0 (Blackwell) |
| Host | `puem`, 16 logical CPUs, Linux 7.0.0-31-generic, glibc 2.43 |
| Python / PyTorch | 3.12.13 / **2.14.0+cu130**, CUDA runtime 13.0 |
| Transformers | 5.17.0 |
| accelerate / bitsandbytes | 1.15.0 / 0.50.2 (`.environment.packages`) |

**`ENV-LANTA`** — LANTA HPC, account `lt200203`
| | |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB, 42,409,000,960 B/ใบ, CC 8.0 (Ampere) |
| Host | `x1001c7s7b0n0` / `lanta-g-175`,`-176`; 64 logical CPUs, Cray Shasta |
| Python / PyTorch | 3.11.10 / **2.7.1+cu126**, CUDA runtime 12.6 |
| Transformers | 5.17.0 |
| accelerate / bitsandbytes | 1.15.0 / 0.50.2 |
| Partition | `gpu-devel` (walltime 2 ชม.) สำหรับ Set A/B/C; `gpu` (5 วัน) สำหรับ Set D |

### 0.2 นิยาม metric และ code ต้นทาง

นิยามที่ freeze ไว้อยู่ใน **`vlm-cxr-benchmark/METRICS.md`** (อยู่ใน
`execution_pack.sha256` และ verify ผ่าน `139ba16b…d9b23`) ตารางด้านล่างคือ
การ map นิยามนั้นไปยังฟังก์ชันที่ implement จริง — ตรวจแล้วว่าตรงกัน เช่น
เกณฑ์ majority `floor(n/2)+1` ใน METRICS.md ตรงกับ `len(rows)//2 + 1` ใน code

| Metric | คำนวณที่ | นิยาม |
|---|---|---|
| `generation_seconds_total` | `vlm_cxr/inference.py` → `summarize_performance()` | ผลรวม `generation_seconds` ของ batch ที่ `warmup=false` **และ** `status=="success"` เท่านั้น |
| `pipeline_seconds_total` | เดียวกัน | ผลรวม `pipeline_seconds` (generation + preprocessing + parsing) เงื่อนไขเดียวกัน |
| `*_throughput_images_per_hour` | เดียวกัน | `3600 × images / *_seconds_total` — **ไม่รวมเวลาโหลดโมเดล** ซึ่งรายงานแยกเป็น `model_loading_seconds` |
| `balanced_accuracy` | `vlm_cxr/metrics.py` → `classification_metrics()` | ค่าเฉลี่ย recall ของ `NORMAL` และ `PNEUMONIA`; `UNCERTAIN` อยู่ในตัวส่วนและนับเป็นผิด |
| `macro_f1` | เดียวกัน | ค่าเฉลี่ย F1 ของสอง true label |
| `uncertainty_rate` | เดียวกัน | สัดส่วนที่ทำนาย `UNCERTAIN` |
| patient-level rows | `vlm_cxr/metrics.py` → `aggregate_patients()` | strict majority (`n//2+1`) ข้ามทุกภาพของผู้ป่วย; เสมอ/พังกลายเป็น `UNCERTAIN` |
| patient bootstrap CI | `vlm_cxr/metrics.py` → `patient_bootstrap_ci()` | 2,000 replicates, seed 20260923, ระดับ 0.95 |

> **`warmup` และ batch ที่ fail ถูกตัดออกจากตัวหารของ throughput ทุกค่า**
> จึงเป็น throughput ของงานที่สำเร็จเท่านั้น ไม่ใช่ operational turnaround
> — แต่ **peak allocated/reserved VRAM นับรวม warm-up ด้วย** ตาม METRICS.md
> จึงเป็นคนละขอบเขตกับตัวเลข throughput ในตารางเดียวกัน

### 0.3 ⚠️ "speedup" ในเอกสารนี้มีสองนิยาม ห้ามเทียบข้ามกัน

| | §1 Experiment A | §3 Experiment C2 |
|---|---|---|
| Code | `tools/summarize_systems_matrix.py` → `paired_speedup_ci()` | เดียวกัน → `replica()` + `main()` |
| สูตร | median ของ ratio รายรอบแบบจับคู่ | `median(1 GPU) / median(n GPU)` |
| ค่ารายรอบ | `pipeline_seconds_total` ของรอบนั้น | `max(rank_pipeline_seconds)` = rank ที่ช้าที่สุด |
| CI | nonparametric bootstrap 2,000 replicates, seed 20260923 | **ไม่มี** |

เลข `0.703×` (§1) กับ `3.914×` (§3) **เป็นคนละชนิดของตัวเลข**

---

## 1. Experiment A — per-GPU speed (batch=1, 1,000 images)

ผลสวนทางกับสมมติฐานตั้งต้น **A100 หนึ่งใบช้ากว่า RTX 5070 ในงานนี้ทั้งสาม profile**

| Profile | Local RTX 5070 | LANTA A100 | A100 speedup | 95% CI | Gate ≥1.5× |
|---|---|---|---|---|---|
| 2b-bf16 | 20,390.6 img/h | 13,271.5 img/h | **0.652×** | [0.635, 0.655] | ❌ fail |
| 4b-bf16 | 15,545.3 img/h | 10,935.0 img/h | **0.703×** | [0.697, 0.714] | ❌ fail |
| 8b-nf4 | 10,254.5 img/h | 8,063.3 img/h | **0.786×** | [0.769, 0.811] | ❌ fail |

> **ที่มา** — ตัวเลข: `runs/pilot-matrix-20260923/systems-summary.json` คีย์
> `experiment_a.<profile>` · คำนวณ: `tools/summarize_systems_matrix.py`
> → `repeated()` + `paired_speedup_ci()` · ต้นทางดิบ:
> `{local,lanta}/A/<profile>/state.json` เขียนโดย `tools/run_repeated_inference.py`
> ซึ่งเรียก `run_stage.py intermediate` → `vlm_cxr/inference.py`
> · launcher: `tools/run_local_a.sh` (local) และ `tools/submit_lanta_matrix.sh` (LANTA)
> · config: `configs/experiments/a-<profile>-batch1.json`
> · environment: **`ENV-LOCAL` เทียบ `ENV-LANTA`, 1 GPU ทั้งคู่**
> · timing scope: `pipeline_seconds_total` (§0.2)

### 1.1 ผลนี้เป็น GPU-bound จริง ไม่ใช่ CPU artifact

เพราะ `pipeline` รวมงาน CPU ไว้ด้วย จึงตรวจซ้ำด้วย generation-only

| Profile | pipeline ratio | generation ratio | ส่วนต่าง |
|---|---|---|---|
| 2b-bf16 | 0.651 | 0.641 | 0.010 |
| 4b-bf16 | 0.703 | 0.699 | 0.004 |
| 8b-nf4 | 0.786 | 0.788 | −0.002 |

สองสเกลแทบเท่ากัน → ช่องว่างเกิดที่ GPU generation ไม่ใช่ preprocessing/parsing

> **ที่มา** — `systems-summary.json` คีย์
> `experiment_a.<profile>.{local_rtx_5070,lanta_a100}.median_{pipeline,generation}_seconds_total`
> · ratio คำนวณสดจาก median ทั้งสองฝั่ง (ไม่ใช่ bootstrap) จึงต่างจากคอลัมน์
> speedup ใน §1 เล็กน้อย

### 1.2 ความถูกต้องของการเปรียบเทียบ

- ✅ **ชุดภาพเดียวกันจริง** — `predictions.jsonl` มี image_id 1,000/1,000 และ
  image_sha256 ตรงกันทั้งหมด (SHA ของ `blind_manifest.csv` ต่างกันเพราะคอลัมน์
  `image_path` เป็น absolute path คนละ root)
- ✅ config เดียวกัน — limit 1000, seed 42, batch_size 1, max_image_edge 512,
  max_new_tokens 128, prompt เดียวกัน, `model_registry_sha256` =
  `d9510493…7445f` ตรงกันทั้งสองฝั่ง
- ✅ peak reserved VRAM ใกล้กันมาก (4.12/8.43/6.23 เทียบ 4.08/8.39/6.26 GiB)
- ⚠️ **software stack ต่างกันโดยจำเป็น** — cu130/Blackwell รันบน A100 ไม่ได้
  จึงเป็นการเทียบ**แพลตฟอร์มตามการใช้งานจริง** ไม่ใช่การแยกผลของ silicon ล้วน ๆ
- ⚠️ **CPU ต่างกันและไม่ได้ควบคุม** — `ENV-LOCAL` มี 16 logical CPUs,
  `ENV-LANTA` มี 64 (จัดสรร 16 cores/GPU) โครงการนี้เคยพิสูจน์แล้วว่า CPU
  partitioning มีผลจริง (ดู §3) แต่ Experiment A ไม่ได้ควบคุมตัวแปรนี้
  **ขนาดและทิศทางของผลกระทบยังไม่ทราบ** — §1.1 แสดงเพียงว่าไม่ได้มาจาก
  CPU-side pipeline overhead
- ⚠️ **code version drift** — ดู §7

### 1.3 ตีความ

ช่องว่างแคบลงตามขนาดโมเดล — local เร็วกว่า 1.53× ที่ 2B, 1.42× ที่ 4B และ 1.27×
ที่ 8B-NF4 สอดคล้องกับการที่ A100 ไล่ทันเมื่อ workload หนักขึ้น และเป็นสัญญาณว่า
ที่ขนาดใหญ่กว่านี้ผลน่าจะกลับด้าน (ยืนยันไม่ได้จากข้อมูลชุดนี้ เพราะ 32B รันบน
local ไม่ได้เลยจึงไม่มีคู่เทียบ)

batch=1 เป็น latency-bound regime ซึ่งเข้าทาง GPU consumer รุ่นใหม่ที่ clock สูง
**ข้อสรุปที่ถูกต้องคือ "ย้ายไป HPC ไม่ได้ช่วยเรื่องความเร็วต่อ GPU"
ไม่ใช่ "HPC ไม่คุ้ม"**

---

## 2. Experiment C1 — batching บน A100 (4b-bf16)

| Batch | Throughput | เทียบ batch1 |
|---|---|---|
| 1 | 10,894.2 img/h | 1.00× |
| 2 | 17,476.0 img/h | 1.60× |
| 4 | 27,540.8 img/h | 2.53× |
| 8 | **41,614.4 img/h** | **3.82×** |

`c1_selected_batch = 8`; prediction status `ok` ครบทุก batch
batch 8 บน A100 เดียว (41,614 img/h) แซง local batch1 (15,545 img/h) ไป 2.68×

> **ที่มา** — `systems-summary.json` คีย์ `c1.<batch>` และ `c1_selected_batch`
> · คำนวณ: `tools/summarize_systems_matrix.py` → `repeated()`; การเลือก batch ใช้
> `median_pipeline_throughput_images_per_hour` สูงสุดในกลุ่มที่
> `status=="complete"` และ `all_predictions_ok`
> · ต้นทางดิบ: `lanta/C1-selected-4b-bf16/batch<N>/state.json`
> · launcher: `tools/submit_selected_c1.sh` → `slurm/repeated-development.sbatch`
> · config: `configs/experiments/c1-4b-bf16-batch<N>.json`,
> manifest `manifests/performance-1000/`
> · environment: **`ENV-LANTA`, 1 GPU** · timing scope: `pipeline_seconds_total`
> · ⚠️ คอลัมน์ "เทียบ batch1" เป็นอัตราส่วน throughput ภายใน C1 เท่านั้น
> ไม่ใช่ speedup แบบ §1 และไม่มี CI

## 3. Experiment C2 — multi-GPU scaling (4b-bf16, CPU-pinned v2)

| GPUs | Median inference pipeline | Speedup | Efficiency | Gate |
|---|---|---|---|---|
| 1 | 176.55 s | 1.000× | 100% | — |
| 2 | 88.69 s | **1.991×** | **99.53%** | ✅ ≥80% |
| 4 | 45.10 s | **3.914×** | **97.86%** | ✅ ≥70% |

scaling เกือบเชิงเส้น ผลนี้ได้หลังแก้ confound เรื่อง CPU contention — รอบก่อนหน้าที่
ไม่ pin CPU ให้แต่ละ rank ทำให้ scaling ดูต่ำผิดจริง เก็บ run เดิมไว้ที่
`lanta/C2-selected-4b-bf16/` และ `…-cpu-pinned/` เป็นหลักฐาน

> **ที่มา** — `systems-summary.json` คีย์ `c2.<gpus>`
> · คำนวณ: `tools/summarize_systems_matrix.py` → `replica()` แล้ว `main()` หาร
> baseline 1 GPU; `scaling_efficiency_percent = 100 × speedup / gpus`;
> threshold 80% (2 GPU) และ 70% (4 GPU)
> · **ค่ารายรอบคือ `max(rank_pipeline_seconds)`** คือ rank ที่ช้าที่สุด ไม่ใช่ค่าเฉลี่ย
> · ต้นทางดิบ: `lanta/C2-selected-4b-bf16-cpu-pinned-v2/gpu<N>/state.json`
> เขียนโดย `tools/run_replica_scaling.py`
> · launcher: `tools/submit_c2_cpu_pinned_v2.sh` — จัดสรร 16 CPU/GPU
> (1→16, 2→32, 4→64 cores) และ pin `rank_cpu_affinities` แยกชุดต่อ rank,
> `torch_num_interop_threads=1`
> · config: `configs/experiments/c2-4b-bf16-batch8.json`,
> manifest `manifests/scaling-2000-selected-4b-bf16/` (**2,000 ภาพ batch 8**
> ไม่ใช่ชุดเดียวกับ §1)
> · environment: **`ENV-LANTA`, 1/2/4 GPU ใน node เดียวกัน**
> · ⚠️ ไม่มี CI — ต่างจาก §1 ดู §0.3
> · JobID `6294204` / `6294205` ใน `slurm-accounting.tsv`

## 4. Local capacity ceiling — 30B และ 32B

รันไม่ได้บน RTX 5070 ตามคาด ซึ่งเป็น **ผลที่ต้องการ** ไม่ใช่การรันล้มเหลว

**หลักฐานหลัก:** `artifacts/local-capacity-32b-no-allocator-warmup.json`
— โหลด BF16 weights จริงลง CUDA device 0 (ไม่มี CPU/disk offload) จนถึง
peak allocated 10.10 GiB / reserved 10.12 GiB จาก 11.50 GiB ที่ใช้ได้ แล้ว OOM
ตอนขอเพิ่มอีกเพียง 80 MiB ทั้งที่ weight รวมทั้งโมเดลคือ 66.71 GB
→ ขาดอีกประมาณ 6 เท่าของความจุการ์ด ต้องใช้ A100 40GB อย่างน้อย 2 ใบ
(`minimum_gpus: 2`)

`30b-a3b-bf16` (MoE, 62.14 GB) ให้ผลเดียวกัน — OOM ทั้งสองโหมด โดยโหมด
no-allocator-warmup โหลดถึง peak reserved 9.83 GiB ก่อน OOM

**ยืนยันจากฝั่งที่รันสำเร็จ (§4b):** การรันจริงบน A100 ใช้ VRAM ต่อ GPU
**30.71 / 31.42 GiB** (32B) และ **28.28 / 30.34 GiB** (30B) จากการ์ดละ 39.5 GiB
เทียบกับ RTX 5070 ที่มีให้ใช้ 11.50 GiB → ต่อให้แบ่ง shard ก็ยังเกินความจุ
ของการ์ดเดียวราว 2.7 เท่า นี่คือเพดานทางกายภาพ ไม่ใช่เรื่องปรับ config

**หลักฐานรอง (อ่อนกว่า):** `artifacts/local-capacity-3Xb.json` — OOM ที่
`caching_allocator_warmup` ของ Transformers ซึ่งขอ 10.30 GiB รวดเดียวตั้งแต่ต้น
เป็น allocator heuristic ไม่ใช่ข้อพิสูจน์ความจุ **ไม่ควรใช้อ้างอิงเป็นหลัก**

> **ที่มา** — `artifacts/local-capacity-{30b,32b}[-no-allocator-warmup].json`
> · คำนวณ/รัน: `tools/probe_local_capacity.py`
> (`probe_policy`: real BF16 model load, all weights forced to CUDA device 0,
> no CPU/disk offload; `skip_allocator_warmup` แยกสองโหมด)
> · checksum: `artifacts/local-3Xb-capacity-evidence.sha256` และ
> `local-3Xb-weight-checksums.sha256`
> · environment: **`ENV-LOCAL` เท่านั้น** (`torch 2.14.0+cu130`, RTX 5070)
> · model revision `0cfaf481…f3ccb` (32B) / `9c4b90e1…1b86c` (30B),
> weight bytes ตรงกับ `configs/models.lock.json`

## 4b. โมเดลใหญ่บน LANTA — Set D, Development 600 ภาพ

**อัปเดต 2026-09-23:** รันเต็มรูปแบบบน Development 600 ภาพชุดเดียวกับ §5 แล้ว
หัวข้อนี้เดิมรายงานผล smoke test 20 ภาพ ซึ่งยังเก็บไว้เป็น audit trail ที่
`outputs/lanta-large-{30b,32b}/` — ตัวเลขด้านล่างคือผลชุดใหม่ที่ **เทียบกับ
`4b-bf16` ได้ตรง** เพราะใช้ `blind_manifest.csv` ไฟล์เดียวกัน
(SHA `2c60565a4764…` ตรงกับที่ Set B ใช้)

| | `30b-a3b-bf16` (MoE) | `32b-bf16` (dense) | `4b-bf16` (comparator) |
|---|---|---|---|
| SLURM JobID | 6295763 | 6295764 | 6293865 |
| n (patients) | 549 | 549 | 549 |
| **Balanced accuracy** | **0.7271** | **0.7335** | 0.6867 |
| 95% CI | [0.6909, 0.7628] | [0.6968, 0.7711] | [0.6517, 0.7216] |
| Macro-F1 | 0.7279 | 0.7436 | 0.6800 |
| Uncertainty | 2.55% | 3.10% | 2.91% |
| recall NORMAL | 0.582 | 0.688 | 0.486 |
| recall PNEUMONIA | 0.873 | 0.779 | 0.888 |
| Model loading (รอบ 1 / รอบ 2) | 59.1 / **33.6** s | 68.3 / **35.6** s | — |
| Generation throughput (รอบ 1 / รอบ 2) | 4,500.0 / 4,647.1 img/h | 5,326.1 / 5,318.5 img/h | — |
| Peak reserved VRAM (2 GPU) | 60.19 GiB | 62.45 GiB | — |
| Wall ทั้ง job (2 รอบ) | 20:13 | 18:18 | — |

**รอบ 1 เป็น cold cache** — elapsed ต่อรอบต่างกันมาก (30B 676.5 → 525.3 s,
32B 615.4 → 469.3 s ราว 29–31%) ต่างจาก Set A ที่รอบต่าง ๆ ห่างกันเพียง 0.6%
แต่เมื่อแยกดูจะเห็นว่า **generation แทบไม่ขยับ** (30B 480.0 → 464.8 s,
32B 405.6 → 406.1 s) ส่วนที่หดคือ model loading ซึ่งลดลงเกือบครึ่ง
(59.1 → 33.6 s และ 68.3 → 35.6 s) กับการอ่าน/verify ภาพจาก Lustre รอบแรก
(`verify_sha256: true` อ่านครบ 600 ไฟล์) **ตัวเลข throughput ที่ควรใช้อ้างอิงคือ
รอบ 2** ส่วนรอบ 1 สะท้อน operational turnaround ตอน cache เย็น

ทั้งสองรัน **600/600 `ok`** ทุกรอบ `exit_code 0` และ
**predictions ของ round-01 กับ round-02 มี SHA-256 ตรงกันเป๊ะทั้งสองโมเดล**
→ ยืนยันว่า generation ที่ batch=1 seed 42 เป็น **deterministic** ซึ่งก่อนหน้านี้
ยังไม่เคยตรวจ

### 4b.1 ผลเทียบแบบจับคู่กับ comparator — นี่คือการทดสอบที่ถูกต้อง

เทียบบนผู้ป่วยชุดเดียวกัน 549 ราย ด้วย paired true-class-stratified patient
bootstrap และ exact McNemar ตาม METRICS.md

| | `30b-a3b-bf16` | `32b-bf16` |
|---|---|---|
| Δ balanced accuracy | **+0.0404** | **+0.0468** |
| 95% CI ของ Δ | [+0.0025, +0.0789] | [+0.0043, +0.0937] |
| McNemar p (exact, two-sided) | 0.0433 | 0.0415 |
| discordant (cand✓/comp✗ : cand✗/comp✓) | 119 (71 : 48) | 176 (102 : 74) |

CI ของ Δ ไม่คร่อมศูนย์ทั้งคู่ และ McNemar p < 0.05 ทั้งคู่ **เมื่อพิจารณาทีละการทดสอบ**
ซึ่งต่างจากตอน n=20 ที่แยกอะไรไม่ได้เลย

> ⚠️ **ข้อควรระวังเรื่อง multiple comparisons — ต้องแจ้งนักสถิติ**
> การทดสอบนี้เอา **candidate 2 ตัวเทียบกับ comparator เดียวกัน** โดยไม่ได้ปรับค่า
> multiplicity ถ้าใช้ Bonferroni สำหรับ 2 การทดสอบ เกณฑ์จะเป็น p < 0.025 ซึ่ง
> **ทั้ง 0.0433 และ 0.0415 ไม่ผ่าน** และ CI ที่ระดับ 97.5% ก็แทบแน่นอนว่าจะคร่อมศูนย์
> เพราะ lower bound ปัจจุบันอยู่ที่ +0.0025 และ +0.0043 เท่านั้น
> เกณฑ์ `McNemar p < 0.05` ที่ freeze ไว้ถูกเขียนสำหรับ candidate **ตัวเดียว**
> การมี 2 ตัวจึงเป็นช่องว่างของ SAP ที่ต้องให้นักสถิติตัดสิน ไม่ใช่สิ่งที่
> engineer ตัดสินเอง
>
> **แต่ข้อนี้ไม่เปลี่ยนคำตัดสินใน §4b.2** — ทั้งคู่ตก Δ ≥ 0.05 อยู่แล้ว
> และ 32B ตกเกณฑ์ recall อยู่แล้ว โดยไม่ต้องพึ่ง p-value

### 4b.2 แต่ทั้งคู่ไม่ผ่าน quality gate ที่ freeze ไว้

เกณฑ์จาก `tools/evaluate_development_matrix.py` → `gate()`
(Δ BA ≥ 0.05, McNemar p < 0.05, ไม่มี class recall ตกต่ำกว่า −0.03)

| Gate | `30b-a3b-bf16` | `32b-bf16` |
|---|---|---|
| Δ BA ≥ 0.05 | ❌ FAIL (0.0404) | ❌ FAIL (0.0468) |
| McNemar p < 0.05 | ✅ PASS | ✅ PASS |
| recall drop ≥ −0.03 | ✅ PASS (−0.015) | ❌ FAIL (−0.109) |

- **30B พลาดเฉียดฉิว** — Δ 0.0404 ขาดอีก 0.0096 จากเกณฑ์ 0.05
  แต่ผ่านเกณฑ์ recall ทั้งสองคลาส
- **32B พลาดสองข้อ** — Δ 0.0468 ขาดอีก 0.0032 และที่สำคัญกว่าคือ
  **แลก recall PNEUMONIA ตกจาก 0.888 เหลือ 0.779 (−0.109)** เพื่อไปได้
  recall NORMAL ดีขึ้น +0.202 ในงานคัดกรองปอดอักเสบ การพลาดเคสป่วยเพิ่มขึ้น
  คือทิศทางที่ผิด เกณฑ์ −0.03 ถูกตั้งมากันเรื่องนี้พอดี

**ข้อสรุป — กรอบที่ถูกต้องของการทดลองนี้:** `4b-bf16` ถูกเลือกจาก
`LOCAL_CANDIDATES = (2b-bf16, 4b-bf16, 8b-nf4)` คือ **โมเดลที่ดีที่สุดเท่าที่รันได้
โดยไม่ต้องใช้ HPC** โดยนิยาม Set D จึงไม่ใช่ข้อเสนอให้เปลี่ยน comparator
แต่เป็นการตอบคำถามตรง ๆ ว่า **"HPC ซื้อคุณภาพให้เราด้วยหรือเปล่า ไม่ใช่แค่ capacity"**

คำตอบคือ **ซื้อได้จริงและวัดได้ แต่ต่ำกว่าเส้นที่โครงการตั้งไว้ล่วงหน้า**
(Δ +0.040 และ +0.047 เทียบเกณฑ์ +0.05) และสำหรับ 32B ยังต้องแลกด้วย
recall PNEUMONIA ที่ตกลง −0.109

นี่คือคู่ขนานฝั่งคุณภาพของข้อค้นพบเรื่องความเร็วใน §1 — HPC ไม่ได้ให้ความเร็ว
ต่อ GPU และให้คุณภาพเพิ่มแบบที่ยังไม่ถึงเกณฑ์ สิ่งที่ HPC ให้แน่ ๆ คือ capacity
และ parallelism หากเจ้าของงานเห็นว่าเกณฑ์ +0.05 เข้มเกินไป นั่นเป็นการตัดสินใจ
ที่ต้องบันทึกพร้อมเหตุผล **ก่อน** เปิด confirmatory ไม่ใช่หลังเห็นตัวเลขนี้

> **ที่มา** — artifacts: `runs/pilot-matrix-20260923/lanta/D-large-development/`
> (`round-01/`, `round-02/`, `evaluation/`) ดึงกลับ local ครบแล้ว
> · quality + paired: `<profile>/evaluation/metrics.json` คีย์
> `primary_patient_level`, `primary_patient_bootstrap_ci`, `paired_comparison`
> · **integrity: `sha256sum -c evaluation/artifact_checksums.sha256` ผ่านทั้งสองชุด
> 0 mismatch**
> · perf/VRAM: `<profile>/round-01/intermediate/summary.json`
> · คำนวณ: `tools/run_repeated_inference.py` → `run_stage.py intermediate` →
> `vlm_cxr/inference.py`; evaluation เรียก `vlm_cxr.evaluate.evaluate_predictions()`
> → `vlm_cxr/metrics.py` (`aggregate_patients()`, `classification_metrics()`,
> `patient_bootstrap_ci()`, `paired_patient_bootstrap_difference_ci()`,
> `paired_mcnemar()`) 2,000 replicates, seed 20260923
> · config: `configs/experiments/d-development-{30b-a3b,32b}-bf16-batch1.json`
> + `paired-config.json` ที่เติม `comparator_predictions`
> · manifest: `manifests/quality-development-600/blind_manifest.csv`
> SHA `2c60565a4764…` **ไฟล์เดียวกับ Set B**
> · comparator predictions SHA `e9ef6257…af67e`
> (`B-development/4b-bf16/round-01/intermediate/predictions.jsonl`)
> · environment: **`ENV-LANTA`, 2 GPU, partition `gpu`** (ไม่ใช่ `gpu-devel`
> เพราะคิวตัน — ดู §6)
> · ⚠️ `dataset.limit: 600` ใน config **ไม่ได้เป็นตัวกำหนดการเลือกภาพ** —
> `run_repeated_inference.py` รับเฉพาะ `--blind-manifest` ค่า limit เป็นเพียง
> metadata
> · ⚠️ ชุดนี้เก็บนอก `B-development/` โดยเจตนา และ **ไม่ได้ถูกรวมเข้า
> `systems-summary.json`** — 30B/32B ไม่ใช่ candidate ที่ pre-declare ไว้ใน
> comparator-selection rule การวางปนกันจะเปิดช่องให้ re-run
> `evaluate_development_matrix.py` เลือก comparator จากโมเดลที่ไม่ได้ declare
> ซึ่งเป็นความเสี่ยง test-driven selection ที่ BLOCK-02 ป้องกันอยู่
> (README อธิบายไว้ในโฟลเดอร์)

### 4b.3 ผล smoke test เดิม 20 ภาพ — เก็บเป็น audit trail

| | `30b-a3b-bf16` | `32b-bf16` |
|---|---|---|
| n | 20 | 20 |
| Balanced accuracy | 0.700 [0.500, 0.900] | 0.750 [0.550, 0.900] |

เทียบกับผลชุด 600 ภาพแล้วเห็นชัดว่า **n=20 ให้ค่ากลางที่คลาดเคลื่อนและ CI
กว้างจนไร้ประโยชน์** — 32B จาก 0.750 ลงมาเป็น 0.7335 และ 30B จาก 0.700
ขึ้นไปเป็น 0.7271 ส่วนอันดับระหว่างสองโมเดลยังเหมือนเดิมแต่ห่างกันน้อยลงมาก
(0.050 → 0.0064) artifacts อยู่ที่ `outputs/lanta-large-{30b,32b}/`
checksum ผ่าน 0 mismatch


## 5. Quality (Development subset, LANTA Set B)

เลือก comparator ด้วยกฎที่ freeze ไว้ล่วงหน้าบน Development 600 images
จาก official **train** เท่านั้น ไม่แตะ official test

| Profile | Balanced acc | Macro-F1 | Uncertainty |
|---|---|---|---|
| 2b-bf16 | 0.6651 | 0.6391 | 1.82% |
| **4b-bf16** | **0.6867** | 0.6800 | 2.91% |
| 8b-nf4 | 0.5771 | 0.5817 | 4.37% |
| 8b-bf16 | 0.4180 | 0.5057 | 36.43% |

**Selected comparator: `4b-bf16`**

> **ที่มา** — `runs/pilot-matrix-20260923/lanta/B-development/quality-summary.json`
> คีย์ `profile_metrics.<profile>` · ต้นทางต่อ: `…/<profile>/evaluation/metrics.json`
> · คำนวณ: `tools/evaluate_development_matrix.py` →
> `vlm_cxr/evaluate.py` → `vlm_cxr/metrics.py` (`aggregate_patients()` แล้ว
> `classification_metrics()`; ค่าที่แสดงเป็น **patient-level**)
> · selection rule (freeze ก่อนรัน): max patient balanced accuracy → tie-break
> macro-F1 สูงกว่า → uncertainty ต่ำกว่า → declared order `2b-bf16`, `4b-bf16`,
> `8b-nf4` (บันทึกในคีย์ `selection_rule`)
> · launcher: `tools/submit_lanta_matrix.sh`, รอผลด้วย `tools/wait_and_evaluate_b.sh`
> · manifest: `manifests/quality-development-600/`
> · environment: **`ENV-LANTA`, 1 GPU** · JobID `6293864`–`6293867`

หมายเหตุ: balanced accuracy ~0.69 ยังต่ำสำหรับงานคลินิก และ `8b-bf16` มี
uncertainty 36.43% ซึ่งบ่งชี้ปัญหา prompt/parser กับ profile นั้น ไม่ใช่ความสามารถ
ของโมเดลล้วน ๆ — โมเดลเดียวกันที่ NF4 ได้ 0.5771 ที่ uncertainty 4.37%

## 6. Resource accounting

ใช้จริง **9.3200 allocated GPU-hours** จากงบ 24 GPU-hours (รวม diagnostics และ
rerun ทั้งหมด) แยกเป็น 8.0361 GPU-h ของ Set A/B/C1/C2 และ **1.2839 GPU-h ของ
Set D** (JobID 6295763 20:13 และ 6295764 18:18 อย่างละ 2 GPU)
Quota คงเหลือ 4,366.41/115,056.00 SHr (3.80%, shared) ณ ก่อนรัน Set D

**หมายเหตุเรื่อง partition:** Set D ไม่ได้รันบน `gpu-devel` เพราะคิวตัน —
partition นั้นมีเพียง 2 nodes และถูกจองเต็ม SLURM ประเมิน start time ไว้อีก
ราว 85 นาที จึง cancel แล้ว resubmit ไป `gpu` (35 nodes สถานะ mix, pending 15)
ซึ่งเริ่มรันภายใน 10 วินาที งานที่ถูก cancel (6295737/6295738) ไม่เคยเริ่มรัน
จึงไม่คิด GPU-hours

> **ที่มา** — ต้นทางดิบ: `runs/pilot-matrix-20260923/slurm-accounting.tsv`
> (จาก `sacct`) และ `quota-after.txt`
> · ⚠️ **ตัวเลข 8.0361 GPU-hours เป็นค่าที่คำนวณด้วยมือจาก TSV
> ไม่มีสคริปต์ใน `tools/` ที่ผลิตค่านี้** ต่างจาก metric อื่นในเอกสารนี้ที่ทุกค่า
> สืบกลับไปถึงฟังก์ชันได้ หากต้องใช้อ้างอิงควรเขียนสคริปต์ derive แล้ว freeze ผล
> · หน่วย SHr เป็น shared service-hour ไม่ใช่ A100 GPU-hour และไม่มี policy
> แปลงหน่วย — ห้ามแปลงเอง (ดู BLOCK-04)

---

## 7. ⚠️ Code integrity — ข้อจำกัดที่ต้องบันทึก

HPCP **ไม่ใช่ git repository** จึงไม่มี commit ให้ pin เวอร์ชัน code
การอ้างอิงในเอกสารนี้จึงใช้ **ชื่อฟังก์ชัน + ไฟล์** ไม่ใช่เลขบรรทัด

### 7.1 Original execution pack drift และ post-run archive

หลังจบทุก run ได้ regenerate post-run archive แล้ว: `sha256sum -c
artifacts/execution_pack.sha256` ผ่าน 73/73 ไฟล์ และ manifest ปัจจุบันมี SHA-256
`bced27c6fe894d3254aaf2fd1885d384d7eb8e628ef9358411bf79fd3aa575c4` อย่างไร
ก็ตาม archive ใหม่นี้ไม่ลบข้อจำกัดทาง provenance ด้านล่าง เพราะ code เปลี่ยนกลาง
Experiment A จริง จึงยังต้อง freeze pack ใหม่ก่อน confirmatory

การตรวจ pack เดิมก่อน regenerate (2026-09-23) ได้ `FAILED` ที่

| ไฟล์ | ใน pack | ปัจจุบัน | mtime |
|---|---|---|---|
| `configs/models.lock.json` | `a7f60f80…7004` | `d9510493…7445f` | 09-23 01:29 |
| `vlm_cxr/inference.py` | `ff65f58a…33f9` | `f66ef88b…dd41` | 09-23 02:56 |

pack ถูกสร้าง 01:15 ก่อนการแก้ทั้งสองครั้ง ที่เหลืออีก 26 ไฟล์ยัง `OK`

`models.lock.json` ไม่เป็นปัญหา — ค่าปัจจุบัน `d9510493…7445f` ตรงกับ
`model_registry_sha256` ที่ทั้ง local และ LANTA บันทึกไว้ในผลรันจริง
แปลว่าทุก run ใช้เวอร์ชันเดียวกัน และเป็นเวอร์ชันหลังแก้

### 7.2 `vlm_cxr/inference.py` เปลี่ยนกลางชุด Experiment A

mtime 2026-09-23 02:56 ICT = 2026-09-22 19:56 UTC เทียบกับเวลารันจริง

| Run | ช่วงเวลา (UTC) | เทียบกับการแก้ |
|---|---|---|
| LANTA A ทั้ง 3 profile | 18:52 → 19:33 | **ก่อน** ทั้งหมด |
| local 2b-bf16 | 19:32 → 19:48 | **ก่อน** ทั้งหมด |
| local 4b-bf16 | 19:48 → 20:08 | **คร่อม** (round-01 ก่อน, round-02…05 หลัง) |
| local 8b-nf4 | 20:08 → 20:40 | **หลัง** ทั้งหมด |

**สิ่งที่เปลี่ยน** — ตรวจจากฟิลด์ที่โผล่ในผลรัน: การแก้เพิ่มการบันทึก
`torch_num_threads`, `torch_num_interop_threads`, `process_cpu_affinity`
และตั้ง `torch_num_threads=10` (จาก 16 cores) ฝั่ง local

**ผลกระทบวัดได้และเล็กมาก** — local 4b-bf16 คร่อมการแก้พอดี จึงใช้เป็นการทดลอง
ธรรมชาติได้

| | round-01 (code เก่า) | round-02…05 (code ใหม่) |
|---|---|---|
| `pipeline_throughput` | 15,620.6 img/h | 15,476.1 – 15,574.7 img/h |
| `generation_seconds_total` | 211.86 s | 212.73 – 213.79 s |

ไม่มี step change ของนิยาม timing — round-01 ต่างจากช่วงของรอบหลังราว **0.3%**
ซึ่งอยู่ในระดับความผันผวนปกติ (รอบ 02–05 เองกระจาย 0.64%) เทียบกับ ratio
0.65–0.79 ที่รายงานใน §1 แล้วเล็กเกินกว่าจะเปลี่ยนข้อสรุป

ทิศทางของผลกระทบยัง **อนุรักษ์นิยม** ต่อข้อสรุป: code ใหม่จำกัด thread ฝั่ง local
เหลือ 10 จาก 16 cores ทำให้ local ช้าลงเล็กน้อย ขณะที่ LANTA ใช้ code เก่าที่ไม่จำกัด
จึงไม่ได้ทำให้ local ดูเร็วเกินจริง

**สิ่งที่ควรทำก่อน confirmatory:** สร้าง `execution_pack.sha256` ใหม่ และ freeze
ก่อนเริ่มรัน ไม่ใช่ระหว่างรัน

---

## ข้อสรุปเชิงระบบ

1. **HPC ไม่ได้ให้ความเร็วต่อ GPU** สำหรับโมเดล 2B–8B ที่ batch=1 — RTX 5070 เร็วกว่า
   1.53× (`2b-bf16`), 1.42× (`4b-bf16`) และ 1.27× (`8b-nf4`)
   คือ speed gate ≥1.5× fail ทั้งสาม profile
2. **HPC ให้ capacity** — 30B MoE และ 32B dense โหลดบนเครื่อง local ไม่ได้เลย (OOM
   ทั้งคู่) แต่รันจบได้จริงบน 2× A100 ที่ Development 600 ภาพเต็มรูปแบบ
   600/600 `ok` นี่คือข้อที่ local ทำไม่ได้ไม่ว่าจะรอนานแค่ไหน
3. **HPC ให้ parallelism** — 4 GPU scaling 3.914× ที่ efficiency 97.86%
4. **HPC ให้ throughput ผ่าน batching** — batch 8 บน A100 เดียวแซง local batch1 2.68×
5. **HPC ให้คุณภาพเพิ่มได้จริง แต่ยังไม่ถึงเกณฑ์** — ที่ n=549 ทั้ง 30B และ 32B
   ชนะโมเดลที่ดีที่สุดซึ่งรันได้โดยไม่ต้องใช้ HPC (Δ BA +0.0404 และ +0.0468)
   แต่ต่ำกว่าเกณฑ์ +0.05 ที่ freeze ไว้ และ 32B ทำ recall PNEUMONIA ตก −0.109
   ผิดเกณฑ์ −0.03 อีกทั้ง p-value ยังไม่ได้ปรับ multiplicity (§4b.1–4b.2)

เหตุผลที่ควรใช้ LANTA ในงานนี้คือข้อ 2–4 ไม่ใช่ข้อ 1 ส่วนข้อ 5 คือคำตอบว่า
คุณภาพดีขึ้นจริงแต่ยังไม่ผ่านเส้นที่ตั้งไว้เอง
