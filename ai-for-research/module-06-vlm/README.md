# Module 6 output — VLM experiment design

ชุดเอกสารนี้เป็นผลจากการใช้ `bin/modules/module-06.md` และ
`bin/workbooks/module-06-experiment-design-workbook.md` กับ
`VLM_experiment_design.md`

## ไฟล์

- `experiment-design-workbook.md` — workbook ที่เชื่อม claim, hypothesis,
  observation, decision rule และสิ่งที่ยังต้องยืนยัน
- `design-red-team-findings.md` — จุดอ่อนของแบบวิจัยและหลักฐานที่ต้องใช้ปิดแต่ละข้อ
- `design-decision-log.md` — แยกสิ่งที่มีอยู่เดิม สิ่งที่แก้แล้วแบบ provisional
  และการตัดสินใจที่ยังต้องให้เจ้าของงานยืนยัน
- `block-closure-checklist.md` — checklist สำหรับตอบคำถาม แนบหลักฐาน และปิด
  BLOCK-01 ถึง BLOCK-04 ก่อนเปิด confirmatory set
- `dataset-download-audit.md` — หลักฐานจากการดาวน์โหลด pinned dataset, export
  เฉพาะ NORMAL/PNEUMONIA และระบุว่า checklist ข้อใดตอบได้หรือยังตอบไม่ได้
- `block-01-dataset-patient-label.md` — BLOCK-01 ฉบับเต็ม: dataset identity,
  class counts, label provenance, duplicate/token audit, official V2/V3 documentation
  และผลตรวจ patient IDs/split จาก V3
- `block-04-lanta-live-audit.md` — live read-only evidence ของ LANTA partitions,
  A100 topology, CPU/RAM และ storage พร้อมขอบเขตที่ยังต้อง quota/runtime/dry run
- `block-03-permission-privacy-ethics.md` — คำตอบจาก public license/source records,
  local metadata/permission audit และรายการที่ต้องให้ owner/สถาบันรับรอง
- `vlm-cxr-benchmark/` — execution pack ที่ตรึง Qwen3-VL revisions, uv GPU
  environments, blind inference, strict parser, patient-level quality metrics,
  systems-performance metrics, SLURM scripts และ automated tests

ชุดข้อมูลหลักย้ายเป็น official Mendeley V3 แล้ว หลักฐานอยู่ใน
`../../data/mendeley-v3-chest-xray/`: CXR JPEG 5,856 ภาพ, patient-aware manifest,
summary, independent verification และ checksums ดาวน์โหลดเฉพาะช่วง CXR 1.255 GB
จาก archive ทางการ จึงไม่เก็บ OCT หรือ content นอกขอบเขต ส่วน metadata preflight
และ V2/HF comparison เก็บไว้ที่
`../../data/hf-vision-chest-xray-pneumonia/source_docs/mendeley-v3/`

ต้นฉบับ `../../VLM_experiment_design.md` ได้รับการแก้เฉพาะส่วนที่ป้องกัน
confirmatory leakage, กำหนด patient-level analysis, freeze comparator,
แยก allocated/reserved GPU-hours และเปิดสถานะ `TBD` ของทรัพยากร/สถิติ

## สถานะ

มีเพียง local readiness smoke 2 Development images เพื่อพิสูจน์ว่า code ทำงานครบ
ซึ่งไม่ใช่ผลวิจัยหรือ quality estimate และยังไม่ถือว่า design พร้อมเปิด
confirmatory set จนกว่าจะปิดรายการ BLOCK-01 ถึง BLOCK-04 ใน
`design-red-team-findings.md`
