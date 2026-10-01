# Dataset Download and Class-Filtering Audit

วันที่ตรวจ: 2026-09-22

> **Superseded for confirmatory use:** เอกสารนี้เป็น audit trail ของ legacy V2/HF
> export หลังเจ้าของงานเลือก migrate ไป official Mendeley V3 แล้ว canonical data,
> manifest และ verification อยู่ที่ `data/mendeley-v3-chest-xray/` และสรุปใน
> `block-01-dataset-patient-label.md`

## Outcome

ดาวน์โหลด `hf-vision/chest-xray-pneumonia` สำเร็จที่ revision
`c1a67c18df1a52ec332ffc00418f4eaa61a9f2bc` และ export เฉพาะสอง class ที่
experiment design ระบุ:

| Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Train | 1,341 | 3,875 | 5,216 |
| Validation | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |
| **Total** | **1,583** | **4,273** | **5,856** |

- Other-class rows: **0**
- Verified readable JPEGs: **5,856/5,856**
- Extracted image bytes: **1,236,482,806 bytes**
- Exact duplicate-content groups: **30**; retained because this pass preserves the
  pinned source rather than silently changing its population
- Source and metadata checksums: **PASS**

ข้อความ narrative ใน source README เขียน 5,863 images แต่ split metadata และ
pinned Parquet มี 5,856 rows จึงใช้จำนวนจริงจาก Parquet และบันทึก discrepancy ไว้
ใน `summary.json`

## Paths

- Source: `/home/puem/Desktop/HPCP/data/hf-vision-chest-xray-pneumonia/source/`
- Filtered images:
  `/home/puem/Desktop/HPCP/data/hf-vision-chest-xray-pneumonia/filtered/{split}/{class}/`
- Manifest:
  `/home/puem/Desktop/HPCP/data/hf-vision-chest-xray-pneumonia/filtered/manifest.csv`
- Summary:
  `/home/puem/Desktop/HPCP/data/hf-vision-chest-xray-pneumonia/filtered/summary.json`
- Metadata audit:
  `/home/puem/Desktop/HPCP/data/hf-vision-chest-xray-pneumonia/filtered/metadata_audit.json`
- Pilot candidate manifest:
  `/home/puem/Desktop/HPCP/data/hf-vision-chest-xray-pneumonia/filtered/pilot_20_candidate_manifest.csv`
  — legacy V2/HF artifact; superseded by V3 candidate manifests

Manifest SHA-256:
`91f2de2a95b5cb4270eeeed7791b1643886b0c989ada4393654a24ebab73f62a`

## What the download can answer

### BLOCK-01 — answered or partially answered

- Dataset ID, pinned revision, original DOI, source site and public license
- Actual row/image count and counts by split/class
- Exact label mapping: `0=NORMAL`, `1=PNEUMONIA`
- Source-described population, label review and quality-screening process
- File integrity, dimensions, format, duplicate-content hashes and filename tokens
- Row-level source shard/index, original filename, class, split and SHA-256

It cannot close BLOCK-01 because the schema contains only `image` and `label`.
There is no verified `patient_id`. Filename tokens were derived for audit only and
must not be represented as confirmed patient identities. Many tokens appear in more
than one split, but their semantics are not authoritative enough to call this
patient leakage without source/custodian verification.

### BLOCK-02 — candidate evidence only

The legacy V2/HF rule produced 20 candidate rows. The earlier image-only subtraction
to 604 images is superseded because a patient-level design must exclude every image
belonging to a touched patient. Canonical V3 candidate artifacts now give 20 candidate
patients covering 42 test images and a provisional patient-excluded pool of 411
patients/582 images (NORMAL 224, PNEUMONIA 358), conditional on owner confirmation
of the actual touched history.

The download cannot determine what the researcher previously opened, what was tuned,
who accessed labels, or how the best-local comparator was chosen. Therefore no
confirmatory manifest has been frozen yet.

### BLOCK-03 — license and metadata evidence only

- HF and Mendeley identify the dataset as CC BY 4.0.
- The license permits copying and adaptation subject to attribution, license link
  and change indication.
- Embedded-metadata scan found EXIF containers in 1,519 JPEGs; 217 include DateTime,
  Software and technical tags. No claim of de-identification is made.
- Pixel-level burned-in identifiers were not cleared by this audit.
- Privacy/ethics rights, institutional policy and written IRB/data-governance
  determination remain external requirements.

### BLOCK-04 — not answered

Dataset download does not establish live A100 inventory, same-node GPU topology,
partition limits, quota, storage on the cluster, runtime versions or successful dry
run. These require live HPC checks and scheduler evidence.

## Official records

- [Hugging Face dataset record](https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia)
- [Mendeley Data V2 record](https://data.mendeley.com/datasets/rscbjbr9sj/2)
- [CC BY 4.0 deed](https://creativecommons.org/licenses/by/4.0/)

## Reproduction

The deterministic exporter is
`/home/puem/Desktop/HPCP/scripts/export_chest_xray_pneumonia.py`. Reproduction
instructions are in
`/home/puem/Desktop/HPCP/data/hf-vision-chest-xray-pneumonia/README.md`.
