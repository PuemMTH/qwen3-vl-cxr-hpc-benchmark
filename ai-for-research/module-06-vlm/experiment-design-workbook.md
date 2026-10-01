# Experiment Design Workbook — VLM Medical Image Classification

> Design record ตาม Module 6; ไม่ใช่ผลวิจัย ช่องที่ยังไม่รู้ใช้รูปแบบ
> `TBD — owner / source / date` และต้องไม่ให้ AI ปิดรายการแทนผู้รับผิดชอบ

## 1. Intake

- **Track:** A — งานของตนเอง
- **Problem:** ต้องแยกว่าประโยชน์ของ A100 มาจาก hardware speed, ความจุที่ทำให้
  รัน 8B BF16 ได้, batch scaling หรือ multi-GPU scaling โดยไม่เอา queue/staging
  ไปปนกับ inference performance
- **Objective/RQ:** ภายใต้ workload ที่ตรึงไว้ A100 ให้ประโยชน์เหนือ RTX 5070
  เมื่อใดในด้าน speed, capacity/quality และ scaling/turnaround
- **Candidate claim (หนึ่งข้อ):** ภายใต้ model revisions, data manifest, prompt,
  parser และ workload ที่ตรึงไว้ A100 ให้ประโยชน์เชิงปฏิบัติเหนือ RTX 5070
  ก็ต่อเมื่อผ่านเกณฑ์ที่ลงทะเบียนล่วงหน้าด้าน capacity/quality หรือ
  speed/scaling และมี complete turnaround ที่เหมาะกับงานภายใน allocation จริง
- **Intended scope/population/environment:** Qwen3-VL Instruct configurations ที่
  ระบุ, Chest X-ray population/site ตาม dataset manifest ที่ยังต้องตรึง, local
  RTX 5070 และ A100 partition/node ที่ตรวจ inventory แล้ว; ไม่ครอบคลุม clinical
  deployment
- **Evidence already verified in source:** experiment matrix, proposed thresholds,
  timing scopes, initial allocation ceiling และ minimum artifact list
- **Target audience/venue family and verified scope:**
  `TBD — PI / target venue author guidelines / date TBD`
- **Unknowns requiring expert review:** confirmatory sample size/power, patient-level
  analysis plan, label validity, data permission/IRB determination, A100 inventory,
  4-GPU topology, current allocation และ operational turnaround requirement

## 2. Claim-to-hypothesis register

| ID | Type | Hypothesis / necessary condition | Why claim needs it | Observation required | Fail/inconclusive condition | Evidence/status |
|---|---|---|---|---|---|---|
| H-C1 | core | A100 เพิ่ม throughput ของ configuration เดียวกันได้ถึง proposed gate | แยก hardware effect จาก model effect | paired rounds ของ 2B/4B BF16 และ 8B NF4 บน hardware ทั้งสอง | speedup ต่ำกว่า gate หรือ CI คร่อม 1 | วิธีทดลองมีแล้ว; CI method รอ SAP |
| H-C2 | core | 8B BF16 บน A100 เพิ่ม balanced accuracy เทียบ frozen best-local comparator ตาม quality gate | รองรับ capacity/quality claim | paired patient-level predictions บน untouched confirmatory set | effect ต่ำกว่า 5 pp, McNemar ไม่ผ่าน, class recall ลดเกิน margin หรือ run ไม่ครบ | threshold เป็น proposed; sample/SAP ยัง TBD |
| H-C3 | core | Batch และ multi-GPU scaling เพิ่ม throughput ตาม efficiency gate โดยไม่เกิด failure ผิดปกติ | รองรับ scaling claim | fixed-workload rounds ที่ batch 1/2/4/8 และ GPU 1/2/4 | OOM, parse failure, 2-GPU <80% หรือ 4-GPU <70% | design ระบุแล้ว; topology/resource ยัง TBD |
| H-M1 | mechanism | ความต่างของ A เกิดจาก hardware เมื่อ model/config/input/order เท่ากัน | ไม่ให้ attributing speedup ผิดสาเหตุ | frozen hashes, matched rounds, node telemetry และ timing scopes | config/hash ต่าง, cache/order bias หรือ throttling ไม่ถูกควบคุม | เพิ่ม telemetry/order control แล้วแบบ provisional |
| H-M2 | mechanism | ความต่างของ B มาจาก precision/capacity ไม่ใช่ comparator selection หรือ test leakage | จำเป็นต่อ BF16 capacity claim | comparator ที่เลือกจาก validation ก่อนเปิด test และ access log | เลือก comparator หลังเห็น test หรือ confirmatory image เคยใช้ tune | source ถูกแก้ให้ freeze comparator; ต้องมี artifact จริง |
| H-V1 | measurement validity | Label และ patient grouping แทน true clinical class ภายใน scope ได้เพียงพอ | metric ถูกต้องได้ต่อเมื่อ ground truth และหน่วยวิเคราะห์เหมาะสม | provenance, label source/quality, repeated-patient audit | correlated images ถูกนับอิสระหรือ split ไม่ group patient | Official V3 migration verified: 3,081 patient IDs and train/test overlap=0; statistician analysis review and duplicate policy remain open |
| H-V2 | measurement validity | Parser/metric นับ `UNCERTAIN` และ parse failure เป็น error โดยไม่ drop denominator | ป้องกัน metric inflation | parser tests และ recomputation จาก raw outputs | unknown outputs ถูกตัดทิ้งหรือ metric library ignore class ผิดวิธี | IMPLEMENTED LOCALLY — strict parser, failure-preserving join, patient aggregation, bootstrap/McNemar และ 9 tests ใน `vlm-cxr-benchmark/`; confirmatory execution ยังไม่เริ่ม |
| H-B1 | boundary/robustness | ข้อสรุปจำกัดเฉพาะ frozen dataset/site/model revisions/hardware states | ป้องกัน generalization เกิน design | scope statement และ manifests ใน report | claim ขยายเป็น VLM/CXR/clinical use ทั่วไป | scope ระบุแล้ว; venue wording ต้อง audit |
| H-B2 | boundary/robustness | 20 pilot images ไม่เข้าสู่ primary confirmatory analysis | ป้องกัน prompt/test contamination | development flag, untouched manifest และ access log | pilot images อยู่ใน primary หรือไม่มี untouched set เพียงพอ | source ถูกแก้แล้ว; BLOCK-02 จนมี manifest |
| H-S1 | safety/feasibility/compliance | Dataset, model และ compute ใช้ได้ตาม permission/license และ institutional determination | ไม่มีสิทธิ/approval จะรันหรือเผยแพร่ไม่ได้ | permission records, licenses, IRB/data-governance determination | permission pending/denied หรือ identifiable data ไม่มี governance | BLOCK-03 |
| H-S2 | safety/feasibility/compliance | A100 resources และ allocation รองรับ matrix โดยไม่เปลี่ยนข้าม node | feasibility ของ C2/24 GPU-hours | live partition inventory, quota และ dry-run | ไม่มี 4-GPU node, RAM/CPU ไม่พอ หรือ quota ต่ำกว่า ceiling | BLOCK-04 |

### Completeness attack

- **Missing hypothesis/condition:** operational deadline ที่นิยามว่า turnaround
  “เหมาะกับงาน” ยังไม่มี threshold และ owner
- **Circular or untestable statement:** “ดีที่สุด” ใช้ไม่ได้จนกว่าจะ freeze
  selection rule และ candidate set บน validation
- **Resource evidence boundary:** live audit ยืนยัน A100 4 GPUs/node, CPU/RAM,
  partition limits และ storage แล้ว แต่ remaining compute quota, environment lock และ
  current-workload dry run ยังเป็น TBD
- **Potentially missing construct:** หากต้องการพูดว่า “คุ้ม” ในเชิงการเงิน ต้องมี
  cost model เพิ่ม; ตอนนี้ design วัด allocation/เวลา ไม่ได้วัด monetary cost

## 3. Experiment/study specification

| Element | Specification | Linked hypothesis ID | Evidence/rationale | Status/owner |
|---|---|---|---|---|
| Unit of analysis | Intended quality unit: patient; systems: measured run/image-equivalent | H-C1,H-C2,H-C3,H-V1 | revised protocol plus V3 patient manifest | patient IDs verified; statistician must freeze patient-level analysis/resampling |
| Population/sample | AP chest X-rays from retrospective pediatric cohort age 1–5 at the source site; untouched confirmatory subset not yet frozen | H-C2,H-B1,H-B2 | dataset card and pinned manifest | source scope verified; confirmatory population/independence open |
| Sampling/allocation | Official V3 train=development and test=confirmatory candidate; train/test are patient-independent; V3 has no validation; legacy 20-image references must be rebuilt/confirmed against V3 | H-V1,H-B2 | `block-01-dataset-patient-label.md` | patient-grouped development/validation rule, pilot history and sample size TBD |
| Intervention/exposure/phenomenon | hardware/model precision, batch และ GPU count | H-C1,H-C2,H-C3 | experiment matrix | specified |
| Comparator/baseline/negative case | same config across hardware; 8B BF16 vs 8B NF4; frozen best-local | H-C1,H-C2,H-M2 | isolates hardware and capacity | selection rule artifact pending |
| Parameters/settings | model revisions, BF16/NF4, prompt, ≤512 px, 128 tokens, greedy, seed | H-M1,H-M2 | source §2 | values/revisions must be frozen |
| Primary outcome/metric | A: throughput/speedup; B: patient-level balanced accuracy; C2: scaling efficiency | H-C1,H-C2,H-C3 | source §§5–8 | metric code/tests pending |
| Secondary outcome/trade-off | latency p50/p95, accuracy, macro-F1, recall/class, uncertain rate, VRAM/RAM, turnaround | all core | source matrix | specified; multiplicity handling TBD |
| Dataset/data source/version | Official Mendeley Data V3 `10.17632/rscbjbr9sj.3`; 5,856 verified CXR JPEGs; NORMAL 1,583/PNEUMONIA 4,273; manifest SHA-256 `1b4cff13…` | H-V1,H-B1,H-B2,H-S1 | `data/mendeley-v3-chest-xray/{manifest.csv,summary.json,verification.json}` | locally frozen; external data-use review and BLOCK-02/03 remain |
| Environment/context | RTX 5070 local vs LANTA A100 `gpu-devel`/`gpu` | H-C1,H-C3,H-S2 | `block-04-lanta-live-audit.md` | static A100×4/node, CPU/RAM, walltime and storage verified; allocation/runtime/dry run pending |
| Instrument/collection method | inference runner, parser, timing/telemetry logger, SLURM/accounting, evaluation script | H-M1,H-V2,H-S2 | minimum artifacts | version/hash and calibration tests pending |
| Procedure/timing | warm-up then 5 measured performance rounds; deterministic quality pass; four timing scopes | H-C1,H-C2,H-C3 | source §§3–9 | paired-order method in SAP pending |
| Confounder/bias/leakage | pilot/test leakage, patient duplication, comparator selection, cache/order, throttling, node variation | H-M1,H-M2,H-V1,H-B2 | red-team audit | controls drafted; evidence pending |
| Analysis/decision rule | proposed speed, quality and scaling gates | H-C1,H-C2,H-C3 | source §§5–8,11 | thresholds require owner/statistician confirmation |
| Failure/inconclusive rule | OOM/failure recorded; invalid protocol/hash/access => no confirmatory claim | all | source + revised status gate | specified provisionally |
| Target-venue evidence/reporting gap | venue unknown; cannot audit reporting expectation | H-B1 | Module 6 requirement | TBD — PI/venue guidelines/date TBD |

## 4. Instrument and measurement audit

| Construct | Instrument/proxy | Validity/calibration evidence | Reliability/repeatability | Pilot needed | Limitation/action |
|---|---|---|---|---|---|
| CXR class correctness | V3 dataset labels + parser + evaluation script | original paper documents expert grading; per-image adjudication unavailable; patient IDs and split independence verified locally | recompute from raw predictions; use patient-level grouping/resampling | yes | label validity is dataset-specific; do not infer clinical utility beyond this cohort |
| Inference latency | synchronized monotonic CUDA/host timing | empty-overhead and scope tests required | warm-up + 5 paired rounds | yes | distinguish request latency from amortized sec/image under batching |
| Throughput | completed image-equivalents / inference seconds | count reconciliation required | fixed manifests and repeated rounds | yes | repeated images valid only for systems performance |
| Peak VRAM | framework allocated/reserved plus device telemetry | compare logger to device state | per-run reset and capture | yes | framework peak is not total board memory |
| GPU utilization/state | device telemetry + node metadata | command/version and sample interval recorded | continuous samples per run | yes | utilization alone does not prove bottleneck |
| Turnaround | submit/start/end/collection timestamps | reconcile runner, SLURM and output manifest | every submission | yes | queue time is operational, not hardware speed |
| Allocation use | scheduler elapsed allocation time × GPU count | reconcile accounting record | every job | yes | report requested ceiling separately |

## 5. Compliance and feasibility trigger screen

| Trigger area | What in this design triggers review | Status | Responsible authority/person | Required document/action |
|---|---|---|---|---|
| Human participants/IRB | retrospective medical images/labels may be human-subject data | official determination required | institutional IRB/data-governance office | protocol/data description and written determination |
| Personal/sensitive data/privacy | image metadata/patient IDs can be identifiable or linkable | official determination required | data custodian/DPO | de-identification, access, retention and export plan |
| Animal/biological/biosafety | no trigger visible in current design | no trigger identified | PI | reconfirm if data source changes |
| Site/data/device permission | medical dataset and HPC access | approval/permission pending | data owner and HPC operator | dataset license/DUA and allocation approval |
| License/IP/contract | Qwen model license, dataset license and output/publication terms | official determination required | PI/legal or institutional contact | archive applicable license versions and use conditions |
| Physical/psychological/environmental risk | no intervention described; compute/resource use only | no trigger identified | PI/HPC operator | follow HPC policy; reassess if clinical workflow is added |

## 6. Result-response matrix

> ทุกแถวเป็น **สถานการณ์สมมติ — ไม่ใช่หลักฐานวิจัย**

| Outcome state | Hypothesis IDs affected | Permitted conclusion | Forbidden conclusion | Claim status | Next action |
|---|---|---|---|---|---|
| Supports decision rule | H-C1/H-C2/H-C3 ตาม gate ที่ผ่าน | configuration นั้นผ่าน predefined gate ภายใน frozen scope และ observed turnaround | A100/VLM ดีกว่าเสมอ, ปลอดภัยทางคลินิก หรือ generalize ข้าม site/model | supported only within scope | audit artifacts/assumptions and report all failures |
| No difference / below threshold | core gate ที่ไม่ผ่าน | ยังไม่มีหลักฐานเพียงพอสำหรับ advantage ตาม gate | “เท่ากัน” หรือ “ไม่มีผล” โดยไม่มี equivalence design | unsupported for that gate | inspect CI/adequacy; keep comparator frozen; plan new study if justified |
| Opposite effect / harm / trade-off | H-C1/H-C2/H-C3 plus class recall/resource trade-off | A100/configuration แย่กว่าใน observed metric หรือมี trade-off ภายใน scope | อธิบายสาเหตุย้อนหลังโดยไม่มี evidence หรือซ่อน failed class/config | contradicted or narrowed | inspect predeclared diagnostics; report adverse trade-off |
| Inconclusive / invalid measurement or protocol | H-M/H-V/H-B/H-S | protocol/measurement/resource ไม่พอให้ตัดสิน | ตีความ partial run เป็น confirmatory result | indeterminate | stop claim, repair protocol, create new immutable execution |

## 7. Red-team findings

รายละเอียดและลำดับความสำคัญอยู่ใน `design-red-team-findings.md` โดยรายการที่
block confirmatory work คือ BLOCK-01 ถึง BLOCK-04

## 8. Design verdict

- **Verdict:** รอเจ้าของงานเลือกหลังยืนยันรายการ BLOCK; AI ไม่เลือกแทน
  - [ ] design supports the candidate claim in principle
  - [ ] design supports only a narrower claim
  - [ ] design needs major revision
  - [ ] design is blocked pending evidence/approval/resource
- **Narrowest claim currently supportable before new data:** design สามารถสร้าง
  หลักฐาน pilot/readiness สำหรับ configuration และ environment ที่ระบุได้ แต่ยัง
  ไม่รองรับ confirmatory quality หรือ clinical/general claim
- **Evidence for readiness choices:**
  - `supports in principle` ต้องมี frozen SAP/manifests/permissions/live inventory
  - `narrower claim` เหมาะถ้าทำเฉพาะ systems-performance และไม่เปิด quality claim
  - `major revision` เหมาะถ้าไม่มี patient IDs/untouched data หรือ comparator freeze ไม่ได้
  - `blocked` เหมาะหาก permission/allocation ยัง pending จนไม่สามารถ pilot ได้
- **Top three revisions:**
  1. ปิด patient-level manifest, label provenance และ untouched confirmatory set
  2. ให้ statistician ยืนยัน sample size, patient-level SAP, CI/McNemar/margins
  3. ตรวจ permission/IRB determination และ live HPC inventory/allocation
- **External review/approval required:** PI/advisor, statistician, data custodian,
  institutional ethics/data-governance authority และ HPC operator
- **What AI must not invent or decide:** sample size, approval/exemption, label
  validity, target venue, operational deadline, current quota หรือ final verdict
