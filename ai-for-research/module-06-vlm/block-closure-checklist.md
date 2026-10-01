# BLOCK Closure Checklist — VLM Experiment Design

ใช้ checklist นี้ก่อนเปิด confirmatory set หากยังไม่ทราบคำตอบ ให้กรอก:

`TBD — ผู้รับผิดชอบ / แหล่งที่จะตรวจ / วันที่คาดว่าจะทราบ`

> การติ๊กช่องหมายถึงมีคำตอบและเปิดหลักฐานตรวจได้ ไม่ใช่เพียงคาดว่าเงื่อนไขนั้น
> น่าจะผ่าน

สถานะอัตโนมัติจาก dataset audit วันที่ 2026-09-22 อ้างอิง
`block-01-dataset-patient-label.md` และ canonical V3 artifacts ใน
`data/mendeley-v3-chest-xray/`; `dataset-download-audit.md` เป็น legacy V2/HF audit
trail เท่านั้น ช่องที่ยังว่าง
ต้องให้เจ้าของงาน/ผู้เชี่ยวชาญตอบ

อัปเดต 2026-09-23: pilot matrix รันครบทุก set แล้ว ไม่มี job ค้าง — local Set A
ครบ 5 รอบ และพิสูจน์ 32B local OOM แล้ว ผลเชิงระบบทั้งหมดสรุปไว้ที่
`systems-results-summary.md` และ aggregate artifact อยู่ที่
`vlm-cxr-benchmark/runs/pilot-matrix-20260923/systems-summary.json`
รายงานย่ออยู่ที่ `vlm-cxr-benchmark/runs/pilot-matrix-20260923/EXPERIMENT_RESULTS.md`
และ evidence manifest 1,199 ไฟล์มี SHA-256
`bce7df5296a584f1445d2ecae9498b1aa2ec44287fb0a0b5e44fcdfb9f12a9c9`
การอัปเดตรอบนี้แตะเฉพาะ BLOCK-04 ช่องที่มีหลักฐานเครื่องรองรับ ช่องที่ต้องให้คน
ตัดสิน (IRB, PI, นักสถิติ, data custodian, touched history) ยังคงว่างตามเดิม

## BLOCK-01 — Dataset, Patient Grouping และ Label Provenance

### ข้อมูลที่ต้องตอบ

- [x] ระบุชื่อ dataset: Kermany, Zhang, and Goldbaum, Mendeley Data V3
- [x] ระบุ dataset version/revision/date: Version 3, DOI `10.17632/rscbjbr9sj.3`;
      migrated/frozen locally 2026-09-22
- [x] ระบุแหล่งที่มา: Guangzhou Women and Children’s Medical Center และ official
      Mendeley archive `ZhangLabData.zip`
- [x] ระบุจำนวนภาพทั้งหมด: 5,856 verified JPEG
- [x] ระบุจำนวนผู้ป่วยไม่ซ้ำ: 3,081 — train 2,650; test 431
- [x] ยืนยันว่ามี `patient_id` ที่ใช้ group ภาพของคนเดียวกันได้: randomized patient ID
      ตาม filename specification ของ V3 และบันทึกใน `manifest.csv`
- [x] ตรวจแล้วว่าผู้ป่วยหนึ่งคนมีหลายภาพหรือไม่
  - [ ] ไม่มี
  - [x] มี จำนวน/รูปแบบ: 1,330 ผู้ป่วยมีหลายภาพ; สูงสุด 30 ภาพ/ผู้ป่วย
- [x] ยืนยันว่า official train/test แบ่งระดับ patient: overlap = 0; V3 ไม่มี validation
- [x] ตรวจ official Mendeley V3 documentation: นิยาม filename เป็น
      `(disease)-(randomized patient ID)-(image number by this patient)` และระบุว่า
      train/test เป็น independent patients
- [x] ตรวจ V3 archive metadata: 3,081 unique patients; train 2,650, test 431;
      train/test patient overlap = 0
- [x] Freeze dataset version: owner เลือก migrate ไป official V3; V2/HF เดิมเก็บเป็น
      audit trail เท่านั้นและไม่ใช้เป็น confirmatory source
- [x] ระบุทุก normalized class: `NORMAL`, `PNEUMONIA`; filename disease labels
      `BACTERIA`/`VIRUS` map เป็น `PNEUMONIA`; ไม่เก็บ OCT หรือ content อื่น
- [x] ระบุแหล่งกำเนิด label
  - [ ] Radiologist
  - [ ] รายงานรังสี
  - [x] Dataset เดิม
  - [ ] ผู้วิจัยกำหนด
  - [x] อื่น ๆ: diagnoses graded by two expert physicians ตาม original paper
- [x] ระบุผู้ตรวจสอบหรือวิธีตรวจคุณภาพ label: two expert physicians;
      evaluation set checked by a third expert ตาม original paper
- [x] ระบุ inclusion criteria: AP chest X-rays จาก retrospective pediatric cohort
      อายุ 1–5 ปี และเป็นภาพจาก routine clinical care
- [x] ระบุ exclusion criteria: original paper ระบุว่าตัด low-quality/unreadable scans
- [x] ระบุวิธีจัดการ missing/uncertain labels: V3 CXR manifest ไม่มี missing/null/
      uncertain label ใน 5,856 records

### หลักฐานสำหรับปิด BLOCK-01

- [x] มี manifest ที่ประกอบด้วยอย่างน้อย `patient_id`, `image_id`, `label`,
      `split`, `source` และ `development/confirmatory flag`
- [x] บันทึก path ของ manifest:
      `data/mendeley-v3-chest-xray/manifest.csv`
- [x] บันทึก SHA-256 ของ manifest:
      `1b4cff13ee5615de5df283372bc91330cdc8ba4f4b5c2e95d1b878329ed5aff0`
- [x] มีรายงานจำนวน images แยกตาม split/class ใน `summary.json`
- [x] มีรายงานจำนวน unique patients และ images แยกตาม split/class
- [x] ตรวจแล้วว่า patient ไม่ซ้ำข้าม split
- [x] ตรวจ manifest เทียบไฟล์จริง: 5,856/5,856; missing=0; unlisted=0;
      per-image SHA-256 failures=0; JPEG failures=0
- [x] ตรวจ exact duplicate content: 27 groups/56 images; ทุกกลุ่มอยู่ใน patient
      เดียวกันและไม่มี group ข้าม split/class
- [ ] Freeze duplicate handling policy ก่อน confirmatory analysis
- [ ] Data custodian ตรวจ provenance/label แล้ว
- [ ] นักสถิติตรวจหน่วยวิเคราะห์และ patient grouping แล้ว

**Owner:** PI + data custodian + statistician  
**Evidence path/link:** `block-01-dataset-patient-label.md` และ
`data/mendeley-v3-chest-xray/`  
**Review date:** Migration/data audit 2026-09-22; external review TBD

- [ ] **BLOCK-01 CLOSED**

---

## BLOCK-02 — Pilot/Test Leakage และ Frozen Comparator

### ข้อมูลที่ต้องตอบ

- [ ] ระบุ image IDs ของ 20 pilot images — สร้าง candidate ตามกฎ “10 test rows
      แรกต่อ class” แล้ว แต่ต้องให้เจ้าของงานยืนยันว่าเป็นชุดที่ใช้จริง
- [x] บันทึก path ของรายการ pilot candidates:
      `data/mendeley-v3-chest-xray/pilot_20_candidate_manifest.csv`
- [x] Rebuild candidate list จาก canonical V3: 20 images, 20 patients,
      NORMAL=10/PNEUMONIA=10, ทุกแถวอยู่ใน test split; SHA-256
      `925444ed8e74052ec0fdc8908456f5fa7dd7b3e71c16cfab9d46e5c418c11e5f`
- [ ] ระบุสิ่งที่เคยปรับโดยดูผลจาก pilot images
  - [ ] Prompt
  - [ ] Parser
  - [ ] Preprocessing
  - [ ] Model/configuration
  - [ ] Precision/quantization
  - [ ] Comparator
  - [ ] ไม่ได้ปรับสิ่งใด
  - [ ] อื่น ๆ: ______________________________________________________
- [ ] ตรวจว่ามี test images อื่นที่เคยเปิดดู prediction/label หรือไม่
  - [ ] ไม่มี
  - [ ] มี ระบุ IDs/path: _____________________________________________
- [ ] ยืนยันว่า pilot/touched images ถูกติดธง `development`
- [ ] ยืนยันว่า pilot/touched images ถูกตัดออกจาก primary confirmatory set
- [x] คำนวณ provisional untouched set แบบตัดทั้ง patient: หาก 20 candidates คือ
      pilot จริง ต้องตัด 42 images จาก 20 patients ไม่ใช่เพียง 20 images
- [ ] ยืนยันจำนวน untouched patients: candidate = 411 เฉพาะเมื่อ owner ยืนยัน
      pilot/touched list
- [ ] ยืนยันจำนวน untouched images: candidate = 582
      (NORMAL 224/PNEUMONIA 358) เฉพาะเมื่อยืนยันว่า touched patients มีเพียง 20 รายนี้
- [x] สร้าง provisional patient-excluded manifest:
      `data/mendeley-v3-chest-xray/untouched_confirmatory_candidate_manifest.csv`
      SHA-256 `e5ab355a3472b60120298bdb728fab243cc75ab6f2811c9ea4a5fef7c7a61d9f`
- [x] ระบุ split ที่ใช้เลือก best-local comparator: deterministic V3
      Development subset จาก official train เท่านั้น 600 images/549 patients,
      NORMAL=300/PNEUMONIA=300; ไม่เปิด official Test
- [x] ระบุ candidate configurations ก่อนเลือก comparator: `2b-bf16`,
      `4b-bf16`, `8b-nf4` ภายใต้ prompt/parser/preprocessing เดียวกัน
- [x] ระบุ metric และ rule ที่ใช้เลือก comparator: patient-level balanced
      accuracy สูงสุด; tie-break ด้วย macro-F1 สูงกว่า, uncertainty ต่ำกว่า แล้วใช้
      declared order `2b-bf16`, `4b-bf16`, `8b-nf4`; ผล Development เลือก
      `4b-bf16`
- [ ] ระบุผู้ที่มีสิทธิ์เปิด confirmatory labels: ________________________

### หากไม่มี untouched set เพียงพอ

- [ ] เลือกหนึ่งแนวทาง
  - [ ] ลดสถานะงานเป็น pilot/exploratory
  - [ ] สร้าง patient-grouped confirmatory split ใหม่
  - [ ] ใช้ external dataset ใหม่
  - [ ] อื่น ๆ: ______________________________________________________

### หลักฐานสำหรับปิด BLOCK-02

- [ ] มี frozen untouched confirmatory manifest พร้อม checksum — provisional manifest
      มีแล้ว แต่ freeze ไม่ได้จนกว่า owner ยืนยัน touched history
- [ ] มี development/pilot flags ครบ
- [ ] มี confirmatory access log
- [x] Freeze และ execute comparator-selection rule บน Development ก่อนเปิด
      confirmatory results; หลักฐาน
      `vlm-cxr-benchmark/runs/pilot-matrix-20260923/lanta/B-development/quality-summary.json`
- [x] สร้าง candidate frozen execution pack สำหรับ prompt, strict parser,
      preprocessing, model revisions และ config hashes ที่
      `vlm-cxr-benchmark/`; final confirmatory freeze ยังห้ามปลด lock จนกว่า
      owner จะยืนยัน touched history/comparator/SAP
- [ ] PI ยืนยันว่าไม่มี test-driven comparator selection
- [ ] นักสถิติยืนยันว่า untouched set เพียงพอตาม SAP

**Owner:** PI + analysis engineer + statistician  
**Evidence path/link:** `data/mendeley-v3-chest-xray/pilot_20_candidate_manifest.csv`,
`data/mendeley-v3-chest-xray/untouched_confirmatory_candidate_manifest.csv`  
**Review date:** candidate manifests 2026-09-22; touched-history review TBD

- [ ] **BLOCK-02 CLOSED**

---

## BLOCK-03 — Permission, Privacy, License และ Ethics

### ข้อมูลที่ต้องตอบ

- [x] ระบุประเภท dataset
  - [x] Public
  - [ ] Institutional
  - [ ] Private
  - [ ] อื่น ๆ: ______________________________________________________
- [x] ระบุผู้เผยแพร่/ผู้สร้างชุดข้อมูล: Kermany, Zhang และ Goldbaum;
      Mendeley Data V3; source clinical site ตาม dataset record/original paper
- [x] ระบุ license/DUA: CC BY 4.0 —
      `https://creativecommons.org/licenses/by/4.0/`
- [x] License ระบุว่า copy/adapt เพื่อวัตถุประสงค์ใดก็ได้เมื่อทำตามเงื่อนไข
- [x] License ไม่มีข้อห้ามด้าน media/format หรือการประมวลผลบน HPC;
      institutional policy ยังต้องตรวจแยก
- [x] License อนุญาต share/adapt โดยต้อง attribution, link license และระบุการแก้ไข;
      privacy/ethics rights ยังต้องตรวจแยก
- [ ] ยืนยันว่าภาพถูก de-identify แล้ว
- [x] ตรวจ JPEG embedded metadata ของ V3 แล้ว: 1,586/5,856 มี EXIF container;
      217 files มี DateTime/Software และ technical tags; ไม่พบ selected sensitive
      EXIF tag names แต่ยังไม่ได้ตีความ XMP/Photoshop หรือ OCR pixels จึงห้ามติ๊ก
      de-identified
- [ ] ไม่มี patient name, HN หรือ direct identifier เหลืออยู่
- [x] ระบุ current local access: effective access = `puem` เท่านั้นสำหรับ ordinary
      local accounts (`/home/puem` mode `750`; group `puem` มีสมาชิกเพียง `puem`)
- [ ] ระบุผู้ที่ได้รับอนุญาตสำหรับ future HPC/backup/collaborator copies: __________
- [x] ระบุสถานที่จัดเก็บข้อมูล:
      `/home/puem/Desktop/HPCP/data/mendeley-v3-chest-xray/`
- [ ] ระบุ retention/deletion plan: ____________________________________
- [ ] ระบุ backup และ access-control plan: ______________________________
- [x] ตรวจ current local permissions/ancestor/group แล้ว: subtree = `775/664`,
      `/home/puem` = `750`, group `puem` มีสมาชิกเดียวและไม่มี additional ACL;
      unrelated ordinary local users traverse ไม่ได้
- [x] ระบุข้อจำกัดการ copy/export ระดับ license: ต้อง attribution, link CC BY 4.0
      และระบุการแก้ไข; สิทธิ privacy/ethics และนโยบายสถาบันไม่ถูกแทนที่ด้วย license
- [ ] ยื่นขอ IRB/data-governance determination แล้ว
  - [ ] ยังไม่ยื่น
  - [ ] ยื่นแล้ว เลขโครงการ: __________________________________________
  - [ ] สถานะ: _______________________________________________________
- [ ] ระบุหน่วยงานหรือผู้มีอำนาจให้คำวินิจฉัย: __________________________

### หลักฐานสำหรับปิด BLOCK-03

- [x] มีหลักฐาน public license จาก HF/Mendeley source record และ CC BY 4.0
- [ ] มี written IRB/privacy/data-governance determination
- [x] มี embedded-metadata audit:
      `data/mendeley-v3-chest-xray/metadata_audit.json` SHA-256
      `56a9d5d96c7bdb3daefa57b2a3cb305af4efed7cc2a6c2eea4399f8c22724efd`
- [ ] มี de-identification determination รวม pixel-level burned-in identifiers และ
      XMP/Photoshop payload review
- [x] มี current local data-access list และ effective owner-only control (`puem`)
- [ ] มี access list/least-privilege control สำหรับ HPC, backup และ future collaborators
- [ ] มี retention/deletion/export plan ที่ผู้รับผิดชอบอนุมัติ
- [x] ตรวจ official model records แล้ว: Qwen3-VL 2B/4B/8B Instruct เป็น
      Apache-2.0, public, not gated ณ 2026-09-23; candidate revision hashes บันทึกใน
      `block-03-permission-privacy-ethics.md`
- [ ] ไม่มีสถานะ permission/approval pending ที่จำเป็นต่อการทดลอง

**Owner:** PI + data custodian  
**Authority/data owner:** institutional IRB/privacy/data-governance authority TBD  
**Evidence path/link:** Mendeley V3/CC BY 4.0 records และ
`data/mendeley-v3-chest-xray/metadata_audit.json` และ
`block-03-permission-privacy-ethics.md`  
**Review date:** license/metadata audit 2026-09-22; access-control audit 2026-09-23;
institutional review TBD

- [ ] **BLOCK-03 CLOSED**

---

## BLOCK-04 — A100, Partition, Quota และ Dry Run

### ข้อมูลที่ต้องตอบ

- [x] ระบุ cluster: LANTA (`lanta`), account `lt200203`
- [x] ระบุ partition: `gpu-devel` สำหรับ quick/dry run; `gpu` สำหรับ production
- [x] ระบุ GPU model: scheduler live inventory ระบุ `a100`
- [x] ระบุ VRAM ต่อ GPU: NVIDIA A100-SXM4-40GB, 42,409,000,960 bytes จาก
      current workload runtime และ `nvidia-smi` logs
- [x] ระบุจำนวน GPU ต่อ node: 4 A100 GPUs
- [x] Scheduler configuration รองรับการขอ 1 GPU ใน node เดียว
- [x] Scheduler configuration รองรับการขอ 2 GPUs ใน node เดียว
- [x] Scheduler configuration รองรับการขอ 4 GPUs ใน node เดียว
- [x] ระบุ CPU ต่อ node: 64; partition default 16 CPU/GPU
- [x] ระบุ RAM ต่อ node: 514,400 MB; partition default 124,000 MB/GPU
- [x] ระบุ walltime limit: `gpu-devel` 2 ชั่วโมง; `gpu` 5 วัน
- [x] ระบุ allocation/quota ที่เหลือหลังจบงาน: 4,366.41/115,056.00 SHr
      (3.80%, shared) จาก `quota-after.txt`; มี warning ต่ำกว่า 10%
- [ ] ยืนยันว่ามี allocation อย่างน้อย 24 A100 GPU-hours — quota รายงานเป็น
      shared SHr ไม่ใช่ guaranteed A100 GPU-hours จึงไม่แปลงหน่วยโดยไม่มี policy;
      งานที่รันจริงรวม Set D/diagnostics/reruns ใช้ 9.3200 allocated GPU-hours
- [x] ระบุพื้นที่ว่างสำหรับงาน: `/lustrefs` ประมาณ 19 TB ณ 2026-09-22 23:44 ICT;
      project ใช้ 36.84/55 TB
- [x] ยืนยันว่ามีพื้นที่อย่างน้อย 150 GB
- [ ] ระบุว่า compute node ออกอินเทอร์เน็ตได้หรือไม่
  - [ ] ได้
  - [ ] ไม่ได้ ต้อง stage model/environment ล่วงหน้า
- [x] งานนี้ไม่พึ่ง compute-node internet: stage model/environment ล่วงหน้าและ
      รัน `--offline --no-sync`; outbound internet policy จริงยังไม่ได้ทดสอบ
- [x] ระบุ Python version: 3.11.10
- [x] ระบุ PyTorch/CUDA versions: PyTorch 2.7.1+cu126, torch CUDA runtime 12.6,
      driver 565.57.01/CUDA capability 12.7
- [x] ระบุ Transformers/Accelerate/quantization versions: Transformers 5.17.0,
      Accelerate 1.15.0, bitsandbytes 0.50.2
- [x] ระบุ source ผู้ยืนยันข้อมูล static: live SLURM `sinfo`/`scontrol`, association
      `teiamarj|lt200203|lanta`; human HPC operator ยังต้องระบุ

### หลักฐานสำหรับปิด BLOCK-04

- [x] มีผล live scheduler/partition inventory พร้อมวันที่ตรวจใน
      `block-04-lanta-live-audit.md`
- [x] มีหลักฐาน current allocation/quota ที่
      `vlm-cxr-benchmark/runs/pilot-matrix-20260923/quota-after.txt`
- [x] มีหลักฐาน free storage
- [x] มี environment manifest/lock file: `vlm-cxr-benchmark/pyproject.toml` และ
      `uv.lock` แยก LANTA A100 (`inference`, cu126) ออกจาก local RTX 5070
      (`local-blackwell`, cu130)
- [x] มี post-run config/source/model revision hash archive ที่
      `vlm-cxr-benchmark/artifacts/execution_pack.sha256`; ตรวจ 73/73 ไฟล์ผ่าน,
      manifest SHA-256 `bced27c6fe894d3254aaf2fd1885d384d7eb8e628ef9358411bf79fd3aa575c4`
      และมี model cache inventories ครบ
      **Audit caveat:** pack เดิมล้าสมัย 2 ไฟล์และ `inference.py` เปลี่ยนกลาง
      Experiment A; local 4B natural check วัดผลต่างราว 0.3% ไม่เปลี่ยนข้อสรุป
      รายละเอียดใน `systems-results-summary.md` §7 แต่ยังต้อง freeze pack ใหม่
      ก่อนเริ่ม confirmatory
- [x] Dry run/current workload บน 1 A100 สำเร็จ
- [x] บันทึก dry-run Job ID: `6293864` (2B Development, 600 images) และ
      matched systems jobs `6293861`–`6293863`
- [x] Dry run ยืนยัน load model, inference, strict parser และ artifact writing;
      600/600 outputs และ 1,000/1,000 outputs ต่อ systems round เป็น `ok`
- [x] local readiness smoke บน RTX 5070 สำเร็จครบ load Qwen3-VL-2B BF16,
      inference, strict parser, patient/image metrics และ artifact writing
      (Development 2 ภาพเท่านั้น; ไม่ใช่หลักฐาน A100/quality)
- [x] บันทึก peak VRAM/host RAM และ OOM trace: runtime summaries มี peak
      allocated/reserved VRAM และ process peak RSS; SLURM `MaxRSS` อยู่ใน
      `slurm-accounting.tsv`; local 30B/32B OOM evidence อยู่ใน `artifacts/` และ
      `logs/`
- [x] 32B local capacity ceiling พิสูจน์แล้ว 2026-09-23 04:17 — หลักฐานหลักคือ
      `artifacts/local-capacity-32b-no-allocator-warmup.json`: โหลด BF16 weights
      จริงลง CUDA device 0 โดยไม่มี CPU/disk offload ถึง peak allocated 10.84 GB /
      reserved 10.86 GB จาก 11.50 GB ที่ใช้ได้ แล้ว OOM ตอนขอเพิ่ม 80 MiB ขณะที่
      weight รวม 66.71 GB (`minimum_gpus: 2`)
      หมายเหตุ: `artifacts/local-capacity-32b.json` เป็นหลักฐานรองที่อ่อนกว่า
      เพราะ OOM ที่ `caching_allocator_warmup` ซึ่งเป็น allocator heuristic
      ไม่ใช่ข้อพิสูจน์ความจุ — ห้ามใช้อ้างอิงเป็นหลัก
- [x] C2 ยืนยัน 2/4 GPUs อยู่ node เดียวกันและแบ่ง disjoint shards ถูกต้อง:
      final jobs `6294204`/`6294205`, 5 rounds, 16 CPU cores/rank,
      `torch_num_interop_threads=1`
- [x] บันทึก queue/start/end/exit status จาก scheduler ใน
      `vlm-cxr-benchmark/runs/pilot-matrix-20260923/slurm-accounting.tsv`
- [x] Large-model feasibility บน LANTA สำเร็จ 2026-09-23 01:39–01:43:
      `30b-a3b-bf16` (JobID 6293741) และ `32b-bf16` (JobID 6293742) รันจบบน
      2× A100-SXM4-40GB, prediction status `ok` 20/20 ทั้งคู่, uncertainty 0%
      config `large-3xb-development.json` ใช้ `source_split: train`,
      `study_role: development`, `limit: 20`, `confirmatory_unlocked: false`
      → เป็น feasibility smoke test ที่ n=20 **ไม่ใช่หลักฐานคุณภาพ** และห้าม
      นำ balanced accuracy 0.700/0.750 ไปเทียบกับ Development 600 images
      artifacts ครบที่ `vlm-cxr-benchmark/outputs/lanta-large-{30b,32b}/`
      (`preprocess/`, `intermediate/`, `final/`);
      `sha256sum -c final/artifact_checksums.sha256` ผ่าน 0 mismatch ทั้งสองชุด
      **ถูกแทนที่แล้วด้วย Set D เต็มรูปแบบ (ด้านล่าง) เก็บไว้เป็น audit trail**
- [x] Set D — 30B/32B รันเต็มรูปแบบบน Development 600 ภาพ 2026-09-23
      JobID `6295763` (30B, 20:13) และ `6295764` (32B, 18:18) partition `gpu`
      2× A100 ต่อ job, REPEATS=2, 600/600 `ok` ทุกรอบ, `exit_code 0`
      artifacts: `runs/pilot-matrix-20260923/lanta/D-large-development/`
      `sha256sum -c evaluation/artifact_checksums.sha256` ผ่าน 0 mismatch ทั้งคู่
      ใช้ `blind_manifest.csv` ไฟล์เดียวกับ Set B (SHA `2c60565a4764…`)
      จึงเทียบกับ `4b-bf16` ได้ตรงบนผู้ป่วย 549 ราย
      **เก็บนอก `B-development/` โดยเจตนา** เพราะ 30B/32B ไม่ใช่ candidate ที่
      pre-declare ไว้ การวางปนจะเปิดช่องให้ re-run เลือก comparator จากโมเดล
      ที่ไม่ได้ declare ซึ่ง BLOCK-02 ป้องกันอยู่ (มี README กำกับในโฟลเดอร์)
- [x] ยืนยัน generation determinism: predictions ของ round-01 กับ round-02 มี
      SHA-256 ตรงกันทั้งสองโมเดล ที่ batch=1 seed 42 (ก่อนหน้านี้ไม่เคยตรวจ)
- [ ] **ผล Set D ไม่ผ่าน quality gate ที่ freeze ไว้ — ต้องให้ PI ตัดสิน**
      30B: Δ BA +0.0404 [+0.0025, +0.0789], McNemar p=0.0433,
      recall delta NORMAL +0.096 / PNEUMONIA −0.015
      → FAIL เกณฑ์ Δ ≥ 0.05 (ขาด 0.0096) แต่ผ่าน recall และ McNemar
      32B: Δ BA +0.0468 [+0.0043, +0.0937], McNemar p=0.0415,
      recall delta NORMAL +0.202 / PNEUMONIA **−0.109**
      → FAIL เกณฑ์ Δ ≥ 0.05 และ FAIL เกณฑ์ recall ≥ −0.03
      ทั้งคู่ดีกว่า comparator เมื่อดูทีละการทดสอบ (CI ของ Δ ไม่คร่อมศูนย์)
      แต่ไม่ถึงเกณฑ์ที่ตั้งไว้ `4b-bf16` คือโมเดลที่ดีที่สุดที่รันได้โดยไม่ต้องใช้ HPC
      อยู่แล้ว ผลนี้จึงตอบว่า **HPC ซื้อคุณภาพเพิ่มได้จริงแต่ยังไม่ถึงเส้นที่
      โครงการตั้งเอง** ไม่ใช่ข้อเสนอให้เปลี่ยน comparator
      หากจะผ่อนเกณฑ์ 0.05 ต้องบันทึกเหตุผล **ก่อน** เปิด confirmatory
- [ ] **Multiplicity ยังไม่ถูกคุม — ต้องให้นักสถิติกำหนดใน SAP**
      เกณฑ์ `McNemar p < 0.05` เขียนไว้สำหรับ candidate ตัวเดียว แต่ Set D เทียบ
      candidate 2 ตัวกับ comparator เดียวกัน ถ้าใช้ Bonferroni (p < 0.025)
      ทั้ง 0.0433 และ 0.0415 ไม่ผ่าน และ CI ที่ 97.5% แทบแน่นอนว่าจะคร่อมศูนย์
      ข้อนี้ไม่เปลี่ยนคำตัดสินของ gate เพราะทั้งคู่ตก Δ ≥ 0.05 อยู่แล้ว
- [x] GPU-hours สะสมหลัง Set D: 9.3200/24 (เดิม 8.0361 + Set D 1.2839)
      บันทึกใน `slurm-accounting.tsv` แล้ว (88 บรรทัด)
- [x] Local Set A ครบ 5 รอบทั้ง 3 profiles (2026-09-23 03:40): `state.json`
      `status: complete`, `exit_code 0` ทุกรอบ, prediction status `ok` ล้วน;
      median elapsed 191.29 / 250.40 / 372.85 s สำหรับ `2b-bf16` / `4b-bf16` /
      `8b-nf4`
- [x] มี systems aggregate artifact ข้าม local/LANTA:
      `vlm-cxr-benchmark/runs/pilot-matrix-20260923/systems-summary.json` (+ `.csv`)
      จาก `tools/summarize_systems_matrix.py` ครอบคลุม Experiment A, C1, C2
- [x] ยืนยันว่า Experiment A เทียบ local กับ A100 บนชุดภาพเดียวกันจริง:
      `predictions.jsonl` มี image_id 1,000/1,000 และ image_sha256 ตรงกันทั้งหมด
      (SHA ของ `blind_manifest.csv` ต่างกันเพราะคอลัมน์ `image_path` เป็น absolute
      path คนละ root เท่านั้น)
- [x] บันทึก caveat ที่แก้ไม่ได้ของ Experiment A: software stack ต่างกันโดยจำเป็น
      (local torch 2.14.0+cu130 บน Blackwell sm_120 เทียบ LANTA torch 2.7.1+cu126
      บน A100 Ampere) จึงเป็นการเทียบแพลตฟอร์มตามการใช้งานจริง ไม่ใช่การแยกผลของ
      silicon ล้วน ๆ

### ต้องให้ PI ตัดสิน (ไม่ใช่ช่องหลักฐาน)

- [ ] **Speed gate ≥1.5× ไม่ผ่านทั้งสาม profile** — A100 หนึ่งใบช้ากว่า RTX 5070
      (speedup 0.652× / 0.703× / 0.786×; 95% CI ไม่คร่อม 1.0 จึงไม่ใช่ noise)
      หลักฐานครบและตรวจได้แล้ว สิ่งที่ยังขาดคือการตัดสินใจว่าจะปรับ gate,
      ปรับ framing ของ claim หรือยอมรับว่าเหตุผลการใช้ HPC คือ capacity และ
      parallelism ไม่ใช่ per-GPU speed — รายละเอียดใน `systems-results-summary.md`

**Owner:** PI + systems lead  
**HPC operator:** human contact TBD; scheduler evidence collected by `teiamarj`  
**Evidence path/link:** `block-04-lanta-live-audit.md`  
**Review date:** live audit 2026-09-22; workload execution 2026-09-23

- [ ] **BLOCK-04 CLOSED**

---

## Final Readiness Gate

- [ ] BLOCK-01 ปิดแล้ว
- [ ] BLOCK-02 ปิดแล้ว
- [ ] BLOCK-03 ปิดแล้ว
- [ ] BLOCK-04 ปิดแล้ว
- [ ] PI ยืนยัน candidate claim
- [ ] นักสถิติยืนยัน sample size และ frozen SAP
- [ ] ยืนยัน proposed thresholds หรือบันทึกค่าที่แก้พร้อมเหตุผล
- [ ] ระบุ operational deadline สำหรับ complete turnaround
- [ ] เลือก final design verdict ใน `experiment-design-workbook.md`
- [ ] เก็บ hashes และ approval/evidence paths ใน `design-decision-log.md`
- [ ] ยังไม่มีผู้ใดเปิด confirmatory results ก่อนผ่าน readiness gate

**Final verdict:** _____________________________________________________  
**Approved by:** _______________________________________________________  
**Date:** ____________________
