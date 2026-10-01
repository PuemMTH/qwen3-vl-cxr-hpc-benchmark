# BLOCK-01 — Dataset, Patient Grouping and Label Provenance

**Status:** `DATA MIGRATION RESOLVED — EXTERNAL REVIEW ITEMS OPEN`  
**Audit date:** 2026-09-22  
**Scope:** official Mendeley Data V3 CXR subset, locally extracted and verified

## 1. Dataset identity and provenance

| Field | Verified value |
|---|---|
| Dataset record | Kermany, Zhang, and Goldbaum (2018), Mendeley Data V3 |
| Frozen version/DOI | Version 3; `10.17632/rscbjbr9sj.3` |
| Official archive | `ZhangLabData.zip`, reported size 8,441,154,111 bytes |
| Official archive SHA-256 | `acbc42869440eec719361942d0ba560f132742230262f22e78bfb93eb5ea4563` |
| Retained scope | `CellData/chest_xray/` only; normalized classes NORMAL/PNEUMONIA |
| Excluded scope | all OCT and non-CXR archive content |
| Selective source range | bytes `0-1255320544` (1,255,320,545 bytes); temporary range removed after verified extraction |
| License stated by Mendeley | CC BY 4.0 |
| Clinical source described by dataset card | Guangzhou Women and Children’s Medical Center, Guangzhou |
| Population described by dataset card | retrospective pediatric cohort อายุ 1–5 ปี |
| Image type | anterior-posterior Chest X-ray, JPEG |
| Local schema | image path, source split, study role, normalized/disease label, documented `patient_id`, image number, dimensions, CRC32 and SHA-256 |

Provenance chain ที่สร้างแล้ว:

```text
Official V3 dataset record + archive identity
  -> selective HTTP byte range containing only CXR records
  -> ZIP central-directory path/size/CRC32 validation
  -> documented randomized patient ID parsed from V3 filename
  -> preserved train/test split + NORMAL/PNEUMONIA mapping
  -> verified JPEG + per-image SHA-256 in manifest.csv
```

ไฟล์ source และ metadata artifacts ผ่าน checksum verification แล้ว

## 2. Counting units and class policy

ต้องแยกหน่วยนับดังนี้:

| Counting stage | Unit | Count | Note |
|---|---|---:|---|
| Official V3 CXR archive entries | image | 5,856 | parsed from official ZIP central directory |
| Locally extracted/readable | verified JPEG | 5,856 | Pillow validation failures = 0 |
| Manifest-to-disk reconciliation | image | 5,856 | missing=0, unlisted=0, SHA-256 failures=0 |
| Target-class export | image | 5,856 | retain only NORMAL/PNEUMONIA CXR paths |
| Excluded archive content | file scope | all | OCT, `.DS_Store` and non-CXR content not retained |

### Verified class mapping

| Normalized class | Train/development | Test/confirmatory | Total |
|---|---:|---:|---:|
| NORMAL | 1,349 | 234 | 1,583 |
| PNEUMONIA | 3,883 | 390 | 4,273 |
| **Total** | **5,232** | **624** | **5,856** |

ไม่มี validation split ใน official V3; train ถูกกำหนดเป็น `development` และ test เป็น
`confirmatory` ใน local manifest โดยยังไม่แบ่ง development เป็น train/validation
ภายใน ไม่มี missing/null/uncertain label การตอบ `UNCERTAIN` ใน experiment เป็น output
ของ model ไม่ใช่ raw dataset class

Official source links checked:

- Mendeley Data V2: <https://data.mendeley.com/datasets/rscbjbr9sj/2>
- Mendeley Data V3: <https://data.mendeley.com/datasets/rscbjbr9sj/3>
- Original Cell paper: <https://doi.org/10.1016/j.cell.2018.02.010>

## 3. Label provenance and quality information

Original paper ระบุว่า:

- ภาพมาจาก routine clinical care ของ pediatric retrospective cohort
- low-quality หรือ unreadable scans ถูกตัดออกก่อน
- diagnoses ถูก graded โดย expert physicians สองคนก่อนใช้ train ระบบ
- evaluation set ถูกตรวจเพิ่มโดย expert คนที่สามเพื่อลด grading error

ข้อมูลนี้เป็น study-level documentation ไม่ใช่ per-image adjudication record
ดังนั้นยังไม่สามารถตรวจ inter-rater agreement, reader identity, report linkage หรือ
adjudication history รายภาพจาก files ที่ดาวน์โหลดได้

## 4. Patient-grouping audit

### Official documentation and migration decision

- คำอธิบายหน้า **V2** ระบุรูปแบบชื่อไฟล์และ independent-patient split เฉพาะ
  `OCT Images`; `code2017.zip/README.md` ก็อธิบายเฉพาะ OCT และใน
  `ChestXRay2017.zip` ไม่มี README/CSV/JSON/TXT/MD ที่ให้ CXR patient mapping
- คำอธิบายหน้า **V3** ขยายข้อความให้ครอบคลุม images ทั้งชุดโดยตรง: แบ่ง train/test
  จากผู้ป่วยคนละกลุ่ม และใช้ชื่อไฟล์รูปแบบ
  `(disease)-(randomized patient ID)-(image number by this patient)`
- เจ้าของงานเลือก migrate ไป V3 เมื่อ 2026-09-22 จึงเลิกใช้ V2/HF เป็น source สำหรับ
  confirmatory experiment และไม่ retrofit V3 patient IDs ลง legacy manifest

ตรวจจากไฟล์ V3 ที่สกัดจริงและ local manifest แล้วได้ผล:

| V3 split/class | Images | Unique patients |
|---|---:|---:|
| Train/NORMAL | 1,349 | 1,008 |
| Train/PNEUMONIA | 3,883 | 1,642 |
| Test/NORMAL | 234 | 229 |
| Test/PNEUMONIA | 390 | 202 |
| **Total** | **5,856** | **3,081** |

- patient overlap ระหว่าง train/test = **0**
- ผู้ป่วยที่มีมากกว่าหนึ่งภาพ = **1,330**; สูงสุด **30** ภาพ/ผู้ป่วย
- ไม่มี patient ที่ข้าม normalized class ภายใน split
- V3 ไม่มี validation split; มี train 5,232 และ test 624 ภาพ

การตรวจ integrity เพิ่มเติมพบ manifest rows=5,856, actual JPEGs=5,856, missing=0,
unlisted=0 และ per-image SHA-256 failures=0 จึงถือว่าการย้ายข้อมูลระดับไฟล์และ
patient grouping เสร็จแล้ว

### Legacy V2/HF comparison retained for audit trail

ก่อน migration พบว่า V3 map exact กลับ V2/HF ได้ 5,788/5,856 rows เท่านั้น มี 68
local rows ที่ map แบบ one-to-one ไม่ได้ และ V3 ไม่มี validation split เหตุผลนี้ทำให้
ต้อง rebuild manifest/pilot references จาก V3 แทนการแก้ legacy manifest

## 5. Duplicate-content audit

V3 พบ exact duplicate-content groups **27 กลุ่ม รวม 56 ภาพ** จาก SHA-256 ของ JPEG
bytes ทั้ง 27 กลุ่มอยู่ภายในผู้ป่วยคนเดียวกัน ไม่มี duplicate group ข้าม patient,
train/test หรือ normalized class ไฟล์ยังถูกเก็บ
ตาม official V3 โดยไม่ลบเงียบ การตัด duplicate หรือเลือกหนึ่งภาพต่อกลุ่มจะเปลี่ยน study
population และต้องเป็น policy ที่ตรึงไว้ก่อน confirmatory analysis

## 6. Evidence artifacts

| Artifact | Path/status |
|---|---|
| Extracted V3 images | `data/mendeley-v3-chest-xray/images/{train,test}/{NORMAL,PNEUMONIA}/` |
| Patient-aware manifest | `data/mendeley-v3-chest-xray/manifest.csv` |
| Manifest SHA-256 | `1b4cff13ee5615de5df283372bc91330cdc8ba4f4b5c2e95d1b878329ed5aff0` |
| Counts/integrity summary | `data/mendeley-v3-chest-xray/summary.json` |
| Independent verification | `data/mendeley-v3-chest-xray/verification.json` |
| Artifact checksums | `data/mendeley-v3-chest-xray/artifact_checksums.sha256` |
| Selective downloader/exporter | `scripts/download_mendeley_v3_cxr.py` |
| Metadata-only V3 preflight | `data/hf-vision-chest-xray-pneumonia/source_docs/mendeley-v3/` |
| Legacy V2/HF data | `data/hf-vision-chest-xray-pneumonia/filtered/` — audit trail only, not confirmatory source |

## 7. What is resolved

- [x] Dataset V3, DOI, official archive identity, source description และ license ถูกบันทึก
- [x] แยกและตรวจเฉพาะ NORMAL/PNEUMONIA
- [x] จำนวนจริงแยกตาม split/class ถูก reconcile กับ official ZIP central directory
- [x] เปิดตรวจ image bytes ครบทุกไฟล์
- [x] มี row-level provenance และ SHA-256
- [x] บันทึก exact duplicate-content groups
- [x] บันทึก label-review process ตาม original paper โดยไม่ขยายเกินหลักฐาน
- [x] ตรวจ official V2/V3 descriptions และ original paper
- [x] ยืนยันความหมาย patient ID และ independent train/test สำหรับ official V3
- [x] คำนวณ V3 unique patients และตรวจ train/test patient overlap
- [x] ตรวจ content identity ระหว่าง current V2/HF กับ V3 ก่อนนำ patient IDs มาใช้
- [x] เจ้าของงานเลือก migrate ไป V3
- [x] ดาวน์โหลด/สกัดเฉพาะ V3 CXR และไม่เก็บ OCT/ไฟล์นอก scope
- [x] สร้าง manifest ที่มี `patient_id`, image/label/source split และ study role
- [x] ตรวจ manifest-to-disk, per-image SHA-256 และ JPEG readability ครบ

## 8. What remains open

- [ ] นักสถิติยืนยันหน่วยวิเคราะห์, resampling/CI และ sample-size plan
- [ ] Freeze policy สำหรับ exact duplicates ก่อนเปิด confirmatory results
- [ ] Data custodian/PI ลงนามรับรอง provenance, license/data-use และ V3 freeze

## 9. Required owner decision

Migration decision เสร็จแล้ว เหลือให้ owner/ผู้เชี่ยวชาญ:

1. นักสถิติ freeze วิธีแบ่ง V3 development เป็น patient-grouped train/validation,
   sample size และ resampling/CI
2. PI/นักสถิติ freeze duplicate policy สำหรับ 27 groups ก่อนเปิด confirmatory result
3. Data custodian/PI ลงนามรับรอง provenance, license/data-use และ dataset freeze

จนกว่าจะมีหลักฐานครบสามข้อ ห้ามติ๊ก `BLOCK-01 CLOSED`; อย่างไรก็ตามข้อ dataset
identity, local files, patient mapping และ train/test patient independence ปิดด้วย
หลักฐานเชิงโปรแกรมแล้ว
