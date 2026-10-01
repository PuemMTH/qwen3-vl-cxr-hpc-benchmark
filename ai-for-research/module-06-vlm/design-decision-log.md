# Design Decision Log

> แยก decision ที่อ่านได้จาก source เดิมออกจาก provisional revision ที่เกิดจาก
> red-team pass นี้ ไม่มีรายการใดเป็นผลการทดลองหรือการอนุมัติจากหน่วยงาน

## Source decisions retained

| ID | Decision | Rationale recorded in source | Status |
|---|---|---|---|
| D-01 | แยก Experiment A/B/C1/C2 | ไม่ปน hardware speed, capacity/quality และ scaling | retained |
| D-02 | ใช้ same model/config สำหรับ hardware effect | ตัด model effect ออกจาก hardware effect | retained |
| D-03 | แยก model loading, inference-only, processing และ turnaround | queue/staging ไม่ใช่ hardware performance | retained |
| D-04 | นับ `UNCERTAIN`/parse failure เป็นผิดใน primary quality analysis | ป้องกัน coverage trade-off ถูกซ่อน | retained and made operationally explicit |
| D-05 | ใช้ fixed workload สำหรับ strong scaling | วัด strong scaling ไม่ใช่ workload growth | retained |
| D-06 | ยอมรับ local/NF4 เป็น valid outcome | design ไม่บังคับให้ HPC ชนะ | retained |

## Provisional revisions from red-team pass

| ID | Before | Provisional revision | Why | Confirmation needed |
|---|---|---|---|---|
| R-01 | pilot images อาจอยู่ใน test ทั้งหมดแล้วใช้ sensitivity analysis | exclude all touched pilot images from primary confirmatory; otherwise label study exploratory | sensitivity analysis ไม่ย้อนคืน test independence | PI/statistician |
| R-02 | image เป็น quality unit | patient เป็น primary unit; group splits/resampling | repeated images from one patient are correlated | statistician/data custodian |
| R-03 | best local comparator ยังไม่มีกระบวนการ freeze | choose candidate/rule from development-validation only and hash before test access | prevent test-driven comparator selection | PI |
| R-04 | balanced accuracy/`UNCERTAIN` behavior implicit | define true-class recall denominator and fixture tests | prevent library-dependent score inflation | analysis engineer/statistician |
| R-05 | GPU-hours = GPUs × reserved walltime | separate allocated elapsed GPU-hours from requested ceiling | planned reservation is not observed consumption | PI/HPC operator |
| R-06 | A100/topology/quota listed as assumptions | record live scheduler evidence separately from still-TBD compute quota/runtime/dry run | static inventory is now verified, but it does not prove workload feasibility or remaining allocation | inventory completed 2026-09-22; HPC operator/PI follow-up remains |
| R-07 | claim scope could be read broadly | explicitly exclude clinical use and out-of-scope populations | design only evaluates frozen dataset/system | PI/venue reviewer |
| R-08 | current V2/HF source ถูกเลือกก่อนพบ official V3 patient mapping | migrate ไป Mendeley V3 และ rebuild manifest จาก official files; ห้าม copy V3 IDs เข้า V2/HF โดยตรง | exact match ได้ 5,788/5,856 rows; 68 local rows map one-to-one ไม่ได้ และ V3 ไม่มี validation split | **owner confirmed 2026-09-22; migration completed** |

## Frozen artifact hashes — pilot matrix 2026-09-23

Artifacts ที่ผลิตเมื่อ pilot matrix รันครบ (local Set A 5 รอบ + 32B capacity probe)
ทั้งหมดอยู่ภายใต้ `vlm-cxr-benchmark/runs/pilot-matrix-20260923/`

| Artifact | SHA-256 |
|---|---|
| `systems-summary.json` | `525f74bf3253259e8cd8cc84f1258073cd4166730b273e2570fb85051b2a7c95` |
| `systems-summary.csv` | `85a0dcf0e8c066a131af79c66f9e720d7689f776c0591c3c6f05b37d4215b048` |
| `lanta/D-large-development/30b-a3b-bf16/evaluation/metrics.json` | `af5a8d763e0b80456caf2db3f7c47bae7ca6a121c48b8dc63eb29d00eead7f9b` |
| `lanta/D-large-development/32b-bf16/evaluation/metrics.json` | `4be94b1599c86d5b787160287952b9c0b1cc227a8905e2977bb8cbfcd41d96f7` |
| `lanta/D-large-development/30b-a3b-bf16/paired-config.json` | `1013ac0e3d00b765775b1165f433f82f15438ada954023dbf88ed7f0b8818425` |
| `lanta/D-large-development/32b-bf16/paired-config.json` | `e7299dd7c3972254a9a220f9680a60cbabc6eeb6743124a5805d1840450c9841` |
| `vlm-cxr-benchmark/artifacts/execution_pack.sha256` | `bced27c6fe894d3254aaf2fd1885d384d7eb8e628ef9358411bf79fd3aa575c4` |
| `experiment-evidence.sha256` (1,199 files) | `bce7df5296a584f1445d2ecae9498b1aa2ec44287fb0a0b5e44fcdfb9f12a9c9` |

Set D (30B/32B บน Development 600, JobID 6295763/6295764) มี
`evaluation/artifact_checksums.sha256` ของตัวเองและ verify ผ่าน 0 mismatch ทั้งคู่
comparator ที่ใช้เทียบคือ `B-development/4b-bf16/round-01/intermediate/predictions.jsonl`
SHA `e9ef625778a3f20090cdd62c5cf9f43326ba5924009f0121b57b98672ddaf67e`

Manifest/dataset hashes ที่ freeze ไว้ก่อนหน้าอยู่ใน `block-closure-checklist.md`
(BLOCK-01/BLOCK-02) และ per-round `summary_sha256` อยู่ใน `state.json` ของแต่ละ profile

Local 32B capacity evidence: `artifacts/local-32b-capacity-evidence.sha256` และ
`artifacts/local-32b-weight-checksums.sha256`

หมายเหตุ: `systems-summary.*` เป็นผล development pilot เท่านั้น
(`confirmatory_test_accessed: false`) ยังไม่ใช่ preregistration bundle ที่
BLOCK-02 ต้องการ — การ freeze ชุดนั้นยังรอ owner ยืนยัน touched history

## Decisions still required from the owner

1. สำหรับ BLOCK-01 ให้นักสถิติ freeze patient-grouped development/validation,
   duplicate policy และ analysis plan; ให้ data custodian/PI ลงนาม data-use/provenance
2. ยืนยัน candidate claim หลัก หรือแก้ให้แคบลงเป็น systems-performance only
3. เลือก final design verdict หนึ่งสถานะใน workbook §8
4. ยืนยัน/แก้ proposed thresholds: speed 1.5×, quality +5 pp,
   recall margin 3 pp, scaling efficiency 80%/70% และ budget 24 GPU-hours
   — **เร่งด่วนขึ้นหลังผล 2026-09-23:** speed gate 1.5× วัดจริงแล้วไม่ผ่านทั้งสาม
   profile (A100 ช้ากว่า RTX 5070; speedup 0.652–0.786×) ส่วน scaling gate
   80%/70% ผ่านสบาย (99.53% / 97.86%) และ budget ใช้จริง 9.32/24 GPU-hours
   **quality gate +5 pp ก็ไม่ผ่านเช่นกัน** — Set D วัดที่ n=549 ได้
   Δ BA +0.0404 (30B) และ +0.0468 (32B) ขาดอีก 0.0096 และ 0.0032 จากเกณฑ์
   และ 32B ยังผิดเกณฑ์ recall margin ด้วย (PNEUMONIA −0.109 เทียบเกณฑ์ −0.03)
   → สองในสาม gate ที่ตั้งไว้ถูกวัดจริงแล้วไม่ผ่าน ควรทบทวนทั้งชุดพร้อมกัน
   ดู `systems-results-summary.md` §1 และ §4b.2
5. ระบุ operational deadline ที่ทำให้ complete turnaround “ตอบโจทย์”
6. ระบุ target venue family เพื่อทำ reporting-gap audit
7. **เกณฑ์ McNemar p < 0.05 เขียนไว้สำหรับ candidate ตัวเดียว** แต่ Set D มี 2 ตัว
   เทียบ comparator เดียวกันโดยไม่ได้ปรับ multiplicity — ถ้าใช้ Bonferroni
   (p < 0.025) ทั้ง 0.0433 และ 0.0415 ไม่ผ่าน และ CI ที่ 97.5% แทบแน่นอนว่า
   จะคร่อมศูนย์ (lower bound ปัจจุบัน +0.0025 และ +0.0043)
   ต้องให้นักสถิติกำหนดวิธีคุม family-wise error ใน SAP ก่อนเปิด confirmatory
   หมายเหตุ: ข้อนี้ไม่เปลี่ยนคำตัดสินของ gate เพราะทั้งคู่ตก Δ ≥ 0.05 อยู่แล้ว

## External action register

| Action | Owner | Source/evidence | Due date |
|---|---|---|---|
| ตรวจรับ V3 patient-level manifest, freeze duplicate/split policy และลงนาม provenance/data-use | PI + data custodian + statistician | `block-01-dataset-patient-label.md` + `data/mendeley-v3-chest-xray/` | TBD; migration itself completed 2026-09-22 |
| Freeze SAP/sample size/CI/test/margins | statistician | signed/hashed SAP | TBD |
| ขอ ethics/privacy/data-use determination | PI + institutional authority | written determination/DUA | TBD |
| ตรวจ A100/4-GPU/CPU/RAM และ quota แบบ live | HPC operator + PI | scheduler inventory/accounting | TBD |
| Freeze comparator/config/parser/source hashes | PI + analysis engineer | preregistration bundle | TBD |

## Final verdict

`TBD — owner / workbook §8 plus closed BLOCK evidence / date TBD`
