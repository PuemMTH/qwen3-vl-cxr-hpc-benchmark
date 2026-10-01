# Design Red-Team Findings

> ผลการตรวจแบบวิจัย ไม่ใช่ผลการทดลอง `VLM_experiment_design.md` ยังอยู่ในสถานะ
> draft pre-registration

## Blocking findings

| ID | Location/element | Finding and logic consequence | Evidence that closes it | Response category | Owner/status |
|---|---|---|---|---|---|
| BLOCK-01 | §3.2; H-V1 | Migration ไป official V3 เสร็จ: 5,856 verified JPEGs, 3,081 patients, train/test overlap=0, patient-aware hashed manifest และ study role มีแล้ว; พบ exact duplicates 27 groups/56 images โดยไม่ข้าม split/class | statistician-approved patient-grouped development/validation and analysis plan; frozen duplicate policy; data-custodian/PI provenance/data-use sign-off | expert/statistician review required | DATA EVIDENCE RESOLVED — EXTERNAL SIGN-OFF OPEN; evidence in `block-01-dataset-patient-label.md` |
| BLOCK-02 | §3.1–3.2; H-B2/H-M2 | V3 pilot and patient-excluded confirmatory candidate manifests สร้างแล้ว; 20 candidate images ครอบคลุม 20 patients/42 total test images จึงเหลือ provisional 411 patients/582 images แต่ยังไม่รู้ว่า IDs เหล่านี้ตรงกับภาพที่เคยดูจริงหรือไม่ และ best-local selection อาจแตะ test | owner-confirmed touched history, frozen primary manifest/flags, validation-only comparator rule and access log | revise now + collect evidence | PARTIALLY RESOLVED — candidate artifacts complete, owner history pending |
| BLOCK-03 | §3.2/compliance; H-S1 | Public CC BY 4.0 record and V3 embedded-metadata audit exist; 1,586 JPEGs contain EXIF containers, selected sensitive EXIF tag names were absent, but XMP/Photoshop values and burned-in pixel identifiers were not cleared and AI cannot declare ethics exemption | written institutional determination plus pixel/XMP review, access and retention/export plan | expert/IRB review required | PARTIALLY RESOLVED — license/aggregate metadata evidence collected; authority/data owner/date TBD |
| BLOCK-04 | §10; H-S2 | Live audit verifies A100×4/node, 64 CPU, 514,400 MB RAM, target partition limits and ~19 TB free storage; uv/model/config/source locks and a local GPU smoke now exist, but remaining compute allocation and current-workload A100 1/2/4-GPU dry runs are still missing | current allocation/quota record, install/verify locked environment on LANTA and immutable A100 dry-run evidence | collect pilot evidence | PARTIALLY RESOLVED — static inventory plus local execution pack; A100 evidence pending |

## High-priority findings

| ID | Location/element | Finding and logic consequence | Evidence that closes it | Response category | Owner/status |
|---|---|---|---|---|---|
| HIGH-01 | §6; H-C2 | confirmatory sample size, power, patient-level bootstrap, multiplicity and McNemar/non-inferiority margins are not frozen | statistician-approved SAP signed/hashed before data access | expert/statistician review required | TBD |
| HIGH-02 | §2/§6; H-V2 | generic metric libraries may mishandle `UNCERTAIN` as a third predicted class or drop parse failures, inflating score | fixture tests proving every uncertain/failure remains in true-class denominator; raw-output recomputation | collect pilot evidence | LOCALLY RESOLVED IN CODE — strict parser/failure denominator/patient aggregation tests pass; confirmatory artifacts pending |
| HIGH-03 | §5; H-M1 | cache state, power/clock/MIG state, node variation and run order can confound hardware speed | paired/alternating schedule, immutable config hashes and per-run telemetry | collect pilot evidence | systems lead |
| HIGH-04 | §7; H-C3 | “latency sec/image” under batching is ambiguous between request latency and amortized processing time | metric dictionary and raw batch start/end timestamps reporting both quantities | revise now | LOCALLY RESOLVED IN CODE — batch latency, amortized sec/image, loading and pipeline scopes are separate; matched A100 runs pending |
| HIGH-05 | §11 | “complete turnaround ตอบโจทย์” ไม่มี operational deadline หรือ threshold จึงใช้ final rule แบบ reproducible ไม่ได้ | owner-approved deadline/service-level criterion frozen before runs | expert review required | TBD — workflow owner/date TBD |

## Scope and reporting findings

| ID | Location/element | Finding and logic consequence | Evidence that closes it | Response category | Owner/status |
|---|---|---|---|---|---|
| SCOPE-01 | Candidate claim | งานประเมิน model/system บน dataset หนึ่ง ไม่รองรับ clinical safety, deployment utility หรือประชากรอื่น | explicit scope in protocol/report and external validation study if later claimed | revise now | scope disclaimer added |
| SCOPE-02 | §8 | reserved walltime ไม่ใช่ consumed GPU-hours; ใช้ค่า ceiling เป็น cost ทำให้ efficiency เพี้ยน | scheduler elapsed allocation records; report reserved ceiling separately | revise now | formula revised |
| SCOPE-03 | §10 | 24 GPU-hours คือ initial reservation budget ไม่ใช่ measured use หรือ monetary cost | planned-vs-actual table and defined cost construct | revise now | terminology clarified |
| SCOPE-04 | Module 6 venue check | ยังไม่ระบุ target venue จึงตรวจ reporting expectation ไม่ได้ | verified venue author/method/reporting guidance | expert review required | TBD — PI/date TBD |

## Revision made in this pass

อย่างน้อยหนึ่งจุดถูกแก้ใน source design ตาม Module 6:

- **ก่อนแก้:** อนุญาตให้ใช้ test split ทั้งหมดแล้วรายงาน sensitivity analysis ที่ตัด
  20 pilot images ออก
- **หลังแก้:** primary confirmatory set ต้อง untouched; 20 pilot images ถูกตัดจาก
  primary และหากไม่มี untouched data เพียงพอให้ลดสถานะเป็น pilot/exploratory

การแก้อื่นที่บันทึกไว้คือ patient-level unit, frozen validation-only comparator,
metric behavior ของ `UNCERTAIN`, live-resource `TBD` และ allocated-vs-reserved
GPU-hours
