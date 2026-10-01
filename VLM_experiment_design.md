# VLM Medical Image Classification — Experiment Design

เอกสารนี้ออกแบบการทดลองสำหรับปรับ Slides 8–13 ของ `HPC-slides.md` โดยแยกคำถามเรื่องความเร็วของ hardware, ประโยชน์ของความจุหน่วยความจำ และการขยาย batch/GPU ออกจากกัน

> **สถานะเอกสาร:** ร่างก่อนลงทะเบียนการทดลอง ยังไม่มีผลวิจัย ตัวเลข threshold,
> วิธีคำนวณช่วงความเชื่อมั่น และขนาด confirmatory set เป็นข้อเสนอที่ต้องได้รับการ
> ยืนยันจาก PI/อาจารย์ที่ปรึกษาและนักสถิติก่อนเปิดชุด confirmatory

## 1. Objective

ประเมินว่า A100 บน HPC ให้ประโยชน์กับ Qwen3-VL สำหรับการจำแนกภาพ Chest X-ray เมื่อใด โดยตอบคำถามหลักสามข้อ:

1. **Hardware effect:** เมื่อใช้ model, precision, batch และข้อมูลเดียวกัน A100 เร็วกว่า RTX 5070 มากพอที่จะคุ้มกับการย้ายงานหรือไม่
2. **Capacity and quality effect:** การที่ A100 รัน Qwen3-VL 8B BF16 ได้ ทำให้คุณภาพดีกว่า 8B NF4 หรือ model ที่ดีที่สุดบนเครื่อง local หรือไม่
3. **Scaling and cost effect:** การเพิ่ม batch size หรือเพิ่ม A100 จาก 1 เป็น 2 และ 4 ใบ เพิ่ม throughput ได้คุ้มกับ GPU-hours และเวลารอคิวหรือไม่

การที่ local/NF4 เพียงพอเป็นข้อสรุปที่ยอมรับได้ ไม่จำเป็นต้องสรุปว่า HPC ดีกว่า

**Candidate claim หลักสำหรับ Module 6:** ภายใต้ model revisions, ข้อมูล, prompt,
parser และ workload ที่ตรึงไว้ A100 ให้ประโยชน์เชิงปฏิบัติเหนือ RTX 5070 ก็ต่อเมื่อ
ผ่านเกณฑ์ที่ลงทะเบียนล่วงหน้าด้าน capacity/quality หรือ speed/scaling และยังให้
complete turnaround ที่เหมาะกับงานภายใน allocation จริง ข้อสรุปนี้ไม่ครอบคลุม
ความพร้อมใช้ทางคลินิกหรือประชากรนอกชุดข้อมูลที่ระบุ

## 2. Pre-registration and frozen controls

ก่อนเริ่ม confirmatory benchmark ให้บันทึกและไม่เปลี่ยนรายการต่อไปนี้:

- Dataset source: official Mendeley Data V3, DOI `10.17632/rscbjbr9sj.3`;
  local manifest `data/mendeley-v3-chest-xray/manifest.csv` SHA-256
  `1b4cff13ee5615de5df283372bc91330cdc8ba4f4b5c2e95d1b878329ed5aff0`
- Model IDs/revisions: `Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203`,
  `Qwen/Qwen3-VL-4B-Instruct@ebb281ec70b05090aa6165b016eac8ec08e71b17` และ
  `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- Precision/quantization: BF16 หรือ NF4; NF4 ใช้ `bitsandbytes==0.50.2`,
  double quantization และ BF16 compute dtype ตาม frozen model registry
- Prompt และ output schema เดียวกันทุก configuration
- Image preprocessing และขนาดด้านยาวไม่เกิน 512 px
- `max_new_tokens=128`, greedy decoding, batch definition และ random seed
- Environment lock: `transformers==5.17.0`, `accelerate==1.15.0`;
  LANTA A100 profile ใช้ `torch==2.7.1+cu126` ส่วน local RTX 5070 profile ใช้
  `torch==2.14.0+cu130`; driver/node จริงต้องบันทึกซ้ำในทุก run
- Label mapping และ parser version
- รายชื่อ patient IDs/image IDs และลำดับภาพ โดยเก็บ patient grouping ไว้แม้ไฟล์หนึ่งคนมีหลายภาพ
- เกณฑ์เลือก `best local comparator` โดยใช้ development/validation evidence เท่านั้น
- Hash ของ configuration, manifest, prompt, parser และ source revision พร้อม access log ของ confirmatory labels

Candidate execution pack และ lockfile อยู่ที่
`ai-for-research/module-06-vlm/vlm-cxr-benchmark/`; confirmatory config ยังล็อก
`confirmatory_unlocked=false` และห้ามเปลี่ยนจนกว่า readiness gate จะผ่าน

Output schema ให้มีสามคำตอบที่ไม่ขัดกับ metric:

```json
{"label":"NORMAL|PNEUMONIA|UNCERTAIN"}
```

หาก output parse ไม่ได้ ให้นับเป็น `UNCERTAIN` และนับเป็นคำตอบผิดใน primary analysis

## 3. Data protocol

### 3.1 Development/pilot set

- Official V3 train split เป็น development pool: 5,232 images จาก 2,650 patients;
  การแบ่ง train/validation ภายในต้อง group ด้วย `patient_id` และ freeze ก่อนใช้เลือก model
- 20 ภาพเดิม ใช้สำหรับตรวจ environment, prompt, parser และ feasibility เท่านั้น
- ห้ามใช้ผล 20 ภาพนี้สรุป general accuracy
- หากเคยปรับ prompt จากการดูผล 20 ภาพ ให้ระบุว่าเป็น development data

### 3.2 Confirmatory quality set

- Official V3 test split เป็น confirmatory candidate pool: 624 images จาก 431
  patients; audit ยืนยันว่าไม่มี patient ID ซ้ำกับ development pool
- Freeze prompt, parser และ configuration ก่อนเปิดผลชุดนี้
- ชุด primary confirmatory ต้องไม่เคยถูกใช้ดูผล ปรับ prompt เลือก model หรือเลือก comparator มาก่อน
- ภาพ 20 ภาพที่เคยดูใน pilot ให้ติดธง `development` และตัด **ทุกภาพของ patient
  เดียวกัน** ออกจาก primary confirmatory analysis; candidate V3 ปัจจุบันจะตัด 42
  images/20 patients และเหลือ 582 images/411 patients หาก owner ยืนยันว่า touched
  history ตรงกับ candidate list อาจรายงานภาพที่ตัดรวมได้เฉพาะ exploratory sensitivity
  analysis ที่แยกป้ายชัดเจน
- หากไม่มีข้อมูล untouched เหลือเพียงพอตามแผนที่นักสถิติยืนยัน ให้หยุดข้อสรุปเชิง confirmatory และรายงานผลเป็น pilot/exploratory เท่านั้น
- ใช้ image IDs ชุดเดียวกันกับทุก model และทุก hardware
- หน่วยวิเคราะห์หลักเป็น patient; ถ้ามีหลายภาพต่อ patient ต้อง group การแบ่งข้อมูลและการ resample/CI ระดับ patient เพื่อไม่ให้นับข้อมูลสัมพันธ์กันเป็นอิสระ
- บันทึก provenance, inclusion/exclusion, label source/quality, missingness, license/permission และ population/site ของข้อมูลก่อนเปิดผล
- การรันภาพเดิมซ้ำไม่เพิ่ม sample size ทางสถิติ

### 3.3 Systems-performance set

- ใช้ ordered list เดียวกันจำนวน **1,000 image-equivalents ต่อรอบ** สำหรับ Experiment A และ batch scaling
- ใช้ fixed workload **2,000 image-equivalents ต่อรอบ** สำหรับ multi-GPU strong scaling
- ทำ warm-up 1 รอบย่อยอย่างน้อย 50 ภาพ แล้วทำ measured runs 5 รอบ
- อนุญาตให้วนรายการภาพเพื่อให้เวลาทดสอบยาวพอ แต่ข้อมูลที่วนซ้ำใช้วัดเวลาเท่านั้น ห้ามนำไปเพิ่มจำนวนตัวอย่างสำหรับ quality statistics
- แยก performance manifest ออกจาก confirmatory-quality manifest และห้ามตีความ image-equivalents ที่วนซ้ำเป็นหลักฐานด้านคุณภาพหรือ clinical validity

## 4. Experiment matrix

| ID | Question | Hardware | Models/configurations | Batch/GPU | Repeats | Primary outputs |
|---|---|---|---|---|---:|---|
| A | Hardware effect | RTX 5070 vs A100 | 2B BF16, 4B BF16, 8B NF4 | batch 1, GPU 1 | 5 | p50/p95 latency, images/hr, speedup, peak VRAM |
| B | Capacity/quality | A100 และ best local comparator ที่เลือกจาก validation ก่อนเปิด test | 8B BF16 vs 8B NF4; เทียบ frozen best local | batch 1, GPU 1 | 1 deterministic pass | balanced accuracy, accuracy, macro-F1, recall/class, uncertainty |
| C1 | Batch scaling | A100 | model ที่ผ่าน B หรือดีที่สุด | per-GPU batch 1/2/4/8 | 5 | throughput, latency, peak VRAM, GPU-hours |
| C2 | Multi-GPU scaling | A100 node เดียว | configuration ที่เลือกจาก C1 | GPU 1/2/4, fixed workload | 5 | speedup, scaling efficiency, turnaround, GPU-hours |

## 5. Experiment A — Hardware effect

### Controlled comparison

- เปรียบเทียบ model เดียวกันกับ model เดียวกันเท่านั้น
- ใช้ batch 1, GPU 1, prompt, image order และ generation configuration เดียวกัน
- Preload input list เพื่อรายงาน **inference-only time** และรัน pipeline ปกติเพื่อรายงาน **end-to-end processing time** แยกกัน
- วัด model-loading time แยก ไม่รวมไว้ใน per-image latency
- ทำ CUDA synchronization ก่อนเริ่มและหลังจบช่วงจับเวลา
- บันทึก peak allocated/reserved VRAM, GPU utilization, CPU utilization และ host RAM
- บันทึก GPU model/UUID, power limit, clock state, MIG state, driver และ node/partition; สลับหรือจับคู่ลำดับ measured rounds เพื่อไม่ให้ warm cache หรือ temporal load เข้าข้าง hardware ใดด้านหนึ่ง

### Metrics

- `Latency p50/p95 [sec/image]`
- `Throughput [images/hr] = 3600 × number of images / inference seconds`
- `Hardware speedup = local median time / A100 median time`
- 95% confidence interval ของ speedup จาก paired measured rounds โดยต้องตรึงวิธีคำนวณหลังนักสถิติทบทวนและก่อนเปิดผล

### Proposed success rule

A100 ถือว่ามี hardware-speed advantage ที่ใช้งานได้เมื่อ:

- throughput speedup อย่างน้อย **1.5×**, และ
- ขอบล่างของ 95% CI ของ speedup มากกว่า 1.0

หากไม่ผ่าน ให้สรุปว่า A100 ยังไม่มีข้อได้เปรียบด้านความเร็วสำหรับ model/configuration นั้น แม้จะมี VRAM มากกว่า

## 6. Experiment B — Capacity and quality effect

### Comparisons

1. **Precision effect:** 8B BF16 vs 8B NF4 บน A100 เดียวกัน
2. **Usable-platform effect:** 8B BF16 บน A100 vs configuration ที่ได้คุณภาพดีที่สุดบน local
3. รายงาน 2B BF16, 4B BF16 และ 8B NF4 เป็น secondary comparisons แต่ไม่เลือก comparator ใหม่หลังเห็นผล test

### Primary and secondary metrics

- Primary: **balanced accuracy** เพื่อไม่ให้จำนวนภาพแต่ละคลาสครอบงำผล โดยคำนวณจาก recall ของ true classes; prediction ที่เป็น `UNCERTAIN` หรือ parse failure ต้องเป็น false negative ของ true class และห้ามถูกตัดออกจาก denominator
- Secondary: accuracy, macro-F1, recall NORMAL, recall PNEUMONIA, precision, coverage และ uncertain rate
- `UNCERTAIN` และ parse failure นับผิดใน balanced accuracy/accuracy/recall
- รายงาน 95% stratified-bootstrap CI ของ metric และความแตกต่างระหว่าง model
- ใช้ paired McNemar test กับสถานะถูก/ผิดของภาพเดียวกัน

ขนาด confirmatory set, วิธี patient-level bootstrap, alpha/multiplicity และ power ของ
McNemar/non-inferiority margin ต้องมี statistical-analysis-plan ที่นักสถิติยืนยันก่อน
เปิดผล ค่าเหล่านี้ยังเป็น `TBD — statistician / frozen SAP / date TBD`

### Proposed quality gate

8B BF16 ถือว่าให้ capacity benefit เมื่อผ่านทุกข้อ:

1. รันสำเร็จครบ 100% ของ confirmatory images โดยไม่ OOM
2. Balanced accuracy สูงกว่า best local comparator อย่างน้อย **5 percentage points**
3. Paired McNemar `p < 0.05`
4. Recall ของคลาสใดคลาสหนึ่งไม่ลดลงเกิน **3 percentage points**

หากผลต่างไม่ผ่านเกณฑ์ ให้รายงานว่า “ยังไม่มีหลักฐานเพียงพอว่า BF16 บน HPC เพิ่มคุณภาพ” ไม่ใช้คำว่าเท่ากันโดยอัตโนมัติ

## 7. Experiment C1 — Batch scaling

- ใช้ A100 1 ใบและ model/configuration เดียว
- นิยาม batch เป็น **per-GPU batch**
- ทดลอง batch 1, 2, 4 และ 8; หยุดขยายเมื่อ OOM
- ใช้ image list และจำนวนภาพต่อรอบเท่ากัน
- วัด 5 measured runs ต่อ batch หลัง warm-up
- รายงาน throughput, latency p50/p95, peak VRAM และ GPU-hours/1,000 images

เลือก batch สำหรับ C2 จาก batch ที่ throughput สูงสุดโดยไม่ OOM และไม่มี latency/parse failure ผิดปกติ ไม่เลือกจาก quality เพราะ generation configuration และ model ต้องคงเดิม

## 8. Experiment C2 — Multi-GPU strong scaling

- ใช้ model replica หนึ่งชุดต่อ GPU และแบ่ง image IDs เป็น disjoint shards
- ใช้ fixed global workload 2,000 image-equivalents ต่อ measured run
- ใช้ per-GPU batch ที่เลือกจาก C1
- ทดลอง GPU 1, 2 และ 4 ใบในโหนดเดียวกัน
- ไม่มี activation exchange ระหว่าง GPU; communication มีเฉพาะ orchestration และการรวมผล

คำนวณ:

- `Speedup(k) = T1 / Tk`
- `Scaling efficiency(k) = Speedup(k) / k × 100`
- `Allocated GPU-hours = k × elapsed allocation time [hr]`
- `Reserved ceiling GPU-hours = k × requested walltime [hr]` (รายงานแยก ไม่ใช้แทนเวลาที่เกิดขึ้นจริง)
- `Allocated GPU-hours/1,000 images = allocated GPU-hours / images × 1,000`

Proposed scaling gate:

- 2 GPU: scaling efficiency อย่างน้อย **80%**
- 4 GPU: scaling efficiency อย่างน้อย **70%**

หากไม่ผ่าน ให้ตรวจ input pipeline, CPU saturation, shard imbalance, model-loading overhead และ result aggregation ก่อนสรุปว่า GPU scaling ไม่มีประโยชน์

## 9. Timing scopes

ต้องรายงานเวลาแยกอย่างน้อยสี่ระดับ:

1. **Model loading:** เริ่มโหลดจนพร้อม infer
2. **Inference-only:** หลัง input พร้อมจน generation เสร็จ
3. **Processing pipeline:** อ่าน/แปลงภาพ + inference + parse/write result
4. **Complete turnaround:** queue + staging + environment startup + loading + processing + evaluation + result collection

Queue time เป็น observed operational result ไม่ใช่ hardware performance และควรรายงาน timestamp, partition, GPU count และ requested walltime ของแต่ละ submission

## 10. Initial allocatable resource request

ตัวเลข walltime ด้านล่างเป็น reservation ceiling สำหรับรอบแรก ต้องปรับหลัง dry-run:

| Stage | Jobs | GPU/job | CPU/job | RAM/job | Walltime/job | Budgeted GPU-hours |
|---|---:|---:|---:|---:|---:|---:|
| Environment dry-run | 1 | 1 | 8 | 32 GB | 0.5 hr | 0.5 |
| A: three shared configurations | 3 | 1 | 8 | 64 GB | 1.5 hr | 4.5 |
| B: BF16/NF4 quality passes | 2 | 1 | 8 | 64 GB | 0.5 hr | 1.0 |
| C1: batch 1/2/4/8 | 4 | 1 | 8 | 64 GB | 1 hr | 4.0 |
| C2: 1-GPU strong scaling | 1 | 1 | 8 | 64 GB | 3 hr | 3.0 |
| C2: 2-GPU strong scaling | 1 | 2 | 16 | 96 GB | 1.5 hr | 3.0 |
| C2: 4-GPU strong scaling | 1 | 4 | 32 | 128 GB | 1 hr | 4.0 |
| **Measured subtotal** | **13** |  |  |  |  | **20.0** |
| **20% rerun contingency** |  |  |  |  |  | **4.0** |
| **Initial request** |  |  |  |  |  | **24 A100 GPU-hours** |

Shared storage request: **150 GB** สำหรับ model snapshots, environment/cache, dataset, logs และ outputs

สถานะทรัพยากรจาก live audit และ Development/Pilot execution 2026-09-22 ถึง
2026-09-23:

- LANTA `gpu-devel`/`gpu` ระบุ A100 4 ใบต่อ node; current workload ยืนยัน
  NVIDIA A100-SXM4-40GB และรัน 1/2/4 GPUs สำเร็จ
- scheduler configuration รองรับ 1/2/4 GPUs ใน node เดียวกัน
- แต่ละ node มี 64 CPUs และ RAM 514,400 MB; default ต่อ GPU คือ 16 CPUs และ
  RAM 124,000 MB
- shared storage ว่างประมาณ 19 TB ซึ่งมากกว่า request 150 GB
- งานทั้งหมดรวม Set D/diagnostic/rerun ใช้จริง 9.3200 allocated GPU-hours;
  quota snapshot ล่าสุดก่อน Set D เหลือ shared balance 4,366.41/115,056.00 SHr
  (3.80%) แต่ SHr ไม่ถูกแปลงเป็น guaranteed A100 GPU-hours โดยไม่มี cluster policy

หลักฐาน: `ai-for-research/module-06-vlm/block-04-lanta-live-audit.md`

หากไม่มี 4-GPU node ให้หยุด C2 ที่ 2 GPU และไม่เปรียบเทียบข้าม node โดยอ้างว่าเป็นผล 4-GPU แบบเดียวกัน

## 11. Final decision rule

เลือก **A100 + 8B BF16** เมื่อ:

- ผ่าน capacity และ quality gate ของ Experiment B และ
- งานอยู่ภายใน 24 GPU-hour budget และ allocation จริง และ
- complete turnaround ตอบโจทย์เวลาของงานได้ดีกว่า local

การใช้ rule นี้เป็นข้อสรุปเชิง confirmatory ทำได้ต่อเมื่อ dataset permission,
patient-level split, frozen comparator และ statistical-analysis-plan ปิดสถานะ `TBD`
แล้ว หากยังไม่ปิด ให้รายงานได้เพียง readiness/pilot finding

เลือก **A100 + model ที่ดีที่สุดซึ่งไม่ใช่ 8B BF16** เมื่อ quality ของ BF16 ไม่ผ่าน แต่ A100 ผ่าน hardware-speed หรือ scaling gate และ workload มีขนาดใหญ่พอชดเชย queue/staging overhead

เลือก **local RTX 5070** เมื่อ best-local quality ไม่ด้อยกว่าเกณฑ์ที่กำหนด และ A100 ไม่ผ่าน speed/scaling gate หรือ complete turnaround ไม่ดีขึ้น

จุดคุ้ม workload สามารถประมาณจาก:

`N_break-even = fixed HPC overhead / (local sec/image − HPC sec/image)`

โดย fixed overhead รวม queue, staging และ model loading หากตัวส่วนไม่เป็นบวก แสดงว่าไม่มี break-even ด้านเวลาสำหรับ configuration นั้น

## 12. Minimum report artifacts

- Frozen configuration YAML/JSON และ model revisions
- Image manifest พร้อม labels และ development/confirmatory flags
- Patient-group manifest, provenance, inclusion/exclusion, label-source และ permission record
- Raw per-image predictions และ correctness pairs
- Raw per-run timing logs แยกตาม timing scope
- SLURM job IDs, resource requests, queue/start/end timestamps และ exit status
- Peak VRAM/host RAM และ OOM traces
- Summary table พร้อม confidence intervals และ paired test
- Frozen statistical-analysis-plan, confirmatory access log และ hashes ของ config/manifest/prompt/parser/source
- รายงานทั้งผลที่ผ่านและไม่ผ่านเกณฑ์ รวมถึง failed/OOM configurations

## 13. Development/Pilot execution record — 2026-09-23

รัน Experiment A/B/C1/C2 ครบและเก็บ state/logs ไว้ที่
`ai-for-research/module-06-vlm/vlm-cxr-benchmark/runs/pilot-matrix-20260923/`
โดยมีสรุปอ่านง่ายที่ `EXPERIMENT_RESULTS.md` ภายในโฟลเดอร์นั้น

- A: RTX 5070 เร็วกว่า A100 ใน matched batch-1 workload ทั้ง 2B BF16, 4B BF16
  และ 8B NF4; A100 speedup estimates = 0.652, 0.703 และ 0.786 จึงไม่ผ่าน
  hardware-speed gate
- B: frozen Development rule เลือก 4B BF16 (patient balanced accuracy 0.6867)
  ขณะที่ 8B BF16 ได้ 0.4180 และไม่ผ่าน numeric quality checks
- C1: batch 8 ให้ median pipeline throughput สูงสุด 41,614.39 images/hour
- C2: หลัง pin CPU แบบ disjoint และจำกัด inter-op threads ได้ speedup 1.991x
  บน 2 GPUs และ 3.914x บน 4 GPUs; efficiency 99.53% และ 97.86%
- capacity extension: 30B MoE BF16 และ 32B dense BF16 เกิด CUDA OOM จริงบน
  local RTX 5070 หลังตรวจ snapshot ครบ แต่ load/infer สำเร็จบน LANTA 2 A100s
  ทั้ง feasibility jobs `6293741`/`6293742` และ Development 600-image jobs
  `6295763`/`6295764`; ได้ Δ balanced accuracy เทียบ 4B = +0.0404/+0.0468
  แต่ทั้งคู่ไม่ผ่าน frozen +0.05 margin และ 32B ไม่ผ่าน class-recall margin

ผลส่วนนี้เป็น Development/Pilot เท่านั้น ไม่ใช่การเปิด confirmatory Test และไม่ทำให้
BLOCK-01/02/03/04 หรือ SAP approval ปิดโดยอัตโนมัติ
