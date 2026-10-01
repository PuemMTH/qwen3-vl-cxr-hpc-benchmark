# ผล Development/Pilot เฉพาะ Experiment A, B, C1, C2

คัดจากรายงานฉบับเต็มวันที่ 23 กันยายน 2026 เฉพาะ §1–5; ตัดรายการโมเดลที่อยู่นอกขอบเขตการนำเสนอออก ตัวเลขเดิมไม่เปลี่ยน

## 1. ข้อมูล โมเดล และวิธีทดลอง

### 1.1 ชุดข้อมูลที่ใช้จริง

ต้นทางคือ Kermany, Zhang, and Goldbaum Mendeley Data V3 ซึ่ง canonical manifest
มี 5,856 ภาพจาก 3,081 ผู้ป่วย การทดลองทั้งหมดในรายงานนี้ดึงจาก official train
split และแยก labels ออกจาก blind inference manifest

| ชุดทดลอง | ภาพ | ผู้ป่วย | NORMAL | PNEUMONIA | การใช้งาน |
|---|---:|---:|---:|---:|---|
| Performance-1000 | 1,000 | 859 | 500 | 500 | A และ C1 |
| Quality-Development-600 | 600 | 549 | 300 | 300 | B |
| Scaling-2000 | 2,000 | 1,529 | 1,000 | 1,000 | C2 |

Dataset manifest SHA-256 คือ
`1b4cff13ee5615de5df283372bc91330cdc8ba4f4b5c2e95d1b878329ed5aff0`
ทุก config ระบุ `source_split: train`, `study_role: development` และ
`confirmatory_unlocked: false`

### 1.2 โมเดลที่ทดสอบ

| Profile | Model ID | Revision | Precision | สถาปัตยกรรม |
|---|---|---|---|---|
| `2b-bf16` | `Qwen/Qwen3-VL-2B-Instruct` | `89644892e4d85e24eaac8bacfd4f463576704203` | BF16 | dense |
| `4b-bf16` | `Qwen/Qwen3-VL-4B-Instruct` | `ebb281ec70b05090aa6165b016eac8ec08e71b17` | BF16 | dense |
| `8b-nf4` | `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | NF4 | dense |
| `8b-bf16` | `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | BF16 | dense |

### 1.3 สภาพแวดล้อม

| รายการ | Local | LANTA |
|---|---|---|
| GPU | GeForce RTX 5070, 12,343,050,240 bytes | A100-SXM4-40GB, 42,409,000,960 bytes/GPU |
| Python | 3.12.13 | 3.11.10 |
| PyTorch | 2.14.0+cu130 | 2.7.1+cu126 |
| CUDA runtime | 13.0 | 12.6 |
| Transformers | 5.17.0 | 5.17.0 |
| Accelerate | 1.15.0 | 1.15.0 |
| bitsandbytes | 0.50.2 | 0.50.2 |

การเทียบ A จึงเป็นการเทียบแพลตฟอร์มที่ใช้งานได้จริง ไม่ใช่การแยกผลของ GPU
silicon ภายใต้ software stack เดียวกัน

### 1.4 Metric และกฎการนับ

- Primary quality metric: patient-level balanced accuracy
- Metrics เพิ่มเติม: accuracy, macro-F1, uncertainty rate, coverage และ
  precision/recall/F1 แยกคลาส
- Patient prediction: strict majority จากทุกภาพของผู้ป่วย; tie, missing,
  runtime failure และ parse failure กลายเป็น `UNCERTAIN`
- `UNCERTAIN` ไม่ถูกตัดออกจากตัวหารของ accuracy, balanced accuracy หรือ recall
- ช่วงความเชื่อมั่น: patient-stratified bootstrap 2,000 replicates, seed 20260923
- Paired comparison: paired patient bootstrap และ exact two-sided McNemar test
- Throughput ใช้เฉพาะ measured batches ที่สำเร็จและไม่รวม model loading
- Peak VRAM รวม warm-up; model loading, generation, pipeline และ complete job
  elapsed แยกรายงานคนละขอบเขต
- Hardware speedup: local pipeline time / A100 pipeline time
- Multi-GPU speedup: median 1-GPU inference time / median k-GPU inference time

## 2. Experiment A — ความเร็วต่อ GPU

แต่ละ profile รัน 5 รอบแบบ fresh process บนรายการ 1,000 image-equivalents
ลำดับเดียวกัน batch 1 และ GPU 1 ใบ ผลทำนายครบ 1,000/1,000 เป็น status `ok`
ทุกครั้ง

### 2.1 Timing, throughput และ VRAM

ค่าทั้งหมดเป็น median ของ 5 รอบ หน่วยเวลาเป็นวินาที และ VRAM เป็น GiB

| Profile | Platform | Load | Generation total | Pipeline total | Generation img/h | Pipeline img/h | Amortized generation p50 | p95 | Peak allocated | Peak reserved |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2B BF16 | Local | 0.837 | 157.983 | 176.552 | 22,787.3 | 20,390.6 | 0.1579 | 0.2164 | 4.06 | 4.12 |
| 2B BF16 | A100 | 2.983 | 246.584 | 271.257 | 14,599.5 | 13,271.5 | 0.2573 | 0.2949 | 4.04 | 4.08 |
| 4B BF16 | Local | 1.232 | 212.803 | 231.581 | 16,917.0 | 15,545.3 | 0.2194 | 0.2827 | 8.38 | 8.43 |
| 4B BF16 | A100 | 5.639 | 304.437 | 329.219 | 11,825.1 | 10,935.0 | 0.3286 | 0.3766 | 8.36 | 8.39 |
| 8B NF4 | Local | 3.142 | 332.465 | 351.066 | 10,828.2 | 10,254.5 | 0.3438 | 0.4249 | 6.18 | 6.23 |
| 8B NF4 | A100 | 11.723 | 421.769 | 446.466 | 8,535.5 | 8,063.3 | 0.4657 | 0.5297 | 6.15 | 6.26 |

Batch-level pipeline latency medians:

| Profile | Platform | Pipeline latency p50 | Pipeline latency p95 |
|---|---|---:|---:|
| 2B BF16 | Local | 0.1741 | 0.2351 |
| 2B BF16 | A100 | 0.2785 | 0.3235 |
| 4B BF16 | Local | 0.2334 | 0.3008 |
| 4B BF16 | A100 | 0.3472 | 0.4050 |
| 8B NF4 | Local | 0.3593 | 0.4466 |
| 8B NF4 | A100 | 0.4838 | 0.5575 |

Peak process RSS medians ซึ่งเป็น host-memory metric:

| Profile | Local GiB | A100 job GiB |
|---|---:|---:|
| 2B BF16 | 4.83 | 4.69 |
| 4B BF16 | 9.13 | 8.99 |
| 8B NF4 | 17.34 | 17.14 |

### 2.2 Speedup gate

| Profile | A100 speedup | 95% bootstrap CI | เกณฑ์ ≥1.5 และ lower CI >1 | ผล |
|---|---:|---:|---|---|
| 2B BF16 | 0.652 | [0.635, 0.655] | ไม่ผ่าน | Local เร็วกว่า 1.53× |
| 4B BF16 | 0.703 | [0.697, 0.714] | ไม่ผ่าน | Local เร็วกว่า 1.42× |
| 8B NF4 | 0.786 | [0.769, 0.811] | ไม่ผ่าน | Local เร็วกว่า 1.27× |

ข้อสรุปของ A จำกัดอยู่ที่ batch 1 และ model sizes 2B–8B การที่ throughput gap
แคบลงเมื่อโมเดลใหญ่ขึ้นไม่ใช่หลักฐานว่า A100 จะเร็วกว่า ณ ขนาดใดขนาดหนึ่ง

## 3. Experiment B — คุณภาพบน Development

ชุด B มี 600 ภาพจาก 549 ผู้ป่วย กฎเลือก best-local comparator คือ patient-level
balanced accuracy สูงสุด ตามด้วย macro-F1 สูงกว่า, uncertainty ต่ำกว่า และ declared
order `2b-bf16`, `4b-bf16`, `8b-nf4`

### 3.1 Patient-level metrics

ค่าที่อยู่ในวงเล็บคือ 95% bootstrap CI

| Profile | n | Accuracy | Balanced accuracy | Macro-F1 | Uncertainty | Coverage | Image BA |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2B BF16 | 549 | 0.6576 [0.6266, 0.6922] | 0.6651 [0.6344, 0.6988] | 0.6391 [0.6001, 0.6804] | 0.0182 | 0.9818 | 0.6733 |
| **4B BF16** | 549 | **0.6812 [0.6466, 0.7159]** | **0.6867 [0.6517, 0.7216]** | **0.6800 [0.6407, 0.7182]** | 0.0291 | 0.9709 | **0.7000** |
| 8B NF4 | 549 | 0.5738 [0.5337, 0.6120] | 0.5771 [0.5364, 0.6153] | 0.5817 [0.5389, 0.6210] | 0.0437 | 0.9563 | 0.5967 |
| 8B BF16 | 549 | 0.4208 [0.3789, 0.4608] | 0.4180 [0.3761, 0.4579] | 0.5057 [0.4611, 0.5480] | 0.3643 | 0.6357 | 0.4417 |

กฎที่กำหนดไว้เลือก `4b-bf16` ตัวเลขของ 8B BF16 ไม่สนับสนุนสมมติฐานว่า precision
ที่สูงขึ้นจะเพิ่มคุณภาพภายใต้ prompt/parser ชุดนี้ เพราะ uncertainty สูงถึง 36.43%
ในระดับผู้ป่วย

### 3.2 Patient-level per-class metrics

| Profile | Class | Support | Predicted | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| 2B BF16 | NORMAL | 282 | 123 | 0.8943 | 0.3901 | 0.5432 |
| 2B BF16 | PNEUMONIA | 267 | 416 | 0.6034 | 0.9401 | 0.7350 |
| 4B BF16 | NORMAL | 282 | 162 | 0.8457 | 0.4858 | 0.6171 |
| 4B BF16 | PNEUMONIA | 267 | 371 | 0.6388 | 0.8876 | 0.7429 |
| 8B NF4 | NORMAL | 282 | 192 | 0.6667 | 0.4539 | 0.5401 |
| 8B NF4 | PNEUMONIA | 267 | 333 | 0.5616 | 0.7004 | 0.6233 |
| 8B BF16 | NORMAL | 282 | 216 | 0.6759 | 0.5177 | 0.5863 |
| 8B BF16 | PNEUMONIA | 267 | 133 | 0.6391 | 0.3184 | 0.4250 |

95% CI ของ recall:

| Profile | NORMAL recall CI | PNEUMONIA recall CI | Uncertainty CI | Coverage CI |
|---|---:|---:|---:|---:|
| 2B BF16 | [0.3333, 0.4504] | [0.9138, 0.9663] | [0.0073, 0.0310] | [0.9690, 0.9927] |
| 4B BF16 | [0.4255, 0.5426] | [0.8502, 0.9213] | [0.0164, 0.0437] | [0.9563, 0.9836] |
| 8B NF4 | [0.3972, 0.5142] | [0.6441, 0.7566] | [0.0273, 0.0619] | [0.9381, 0.9727] |
| 8B BF16 | [0.4610, 0.5745] | [0.2622, 0.3745] | [0.3242, 0.4044] | [0.5956, 0.6758] |

### 3.3 Paired checks ของ 8B BF16

| Comparator | Δ balanced accuracy | Δ recall NORMAL | Δ recall PNEUMONIA | McNemar discordant | p-value | ผ่าน numeric gate |
|---|---:|---:|---:|---:|---:|---|
| 8B NF4 | −0.1591 | +0.0638 | −0.3820 | 158 | 1.30×10⁻¹¹ | ไม่ผ่าน |
| Best local: 4B BF16 | −0.2687 | +0.0319 | −0.5693 | 249 | 1.80×10⁻²⁰ | ไม่ผ่าน |

ค่า p ที่ต่ำในตารางนี้แสดงความแตกต่างแบบ paired แต่ทิศทางเป็นผลเสียต่อ 8B BF16
จึงไม่ใช่หลักฐานสนับสนุน candidate

### 3.4 Runtime ของ quality pass

Experiment B รัน deterministic pass หนึ่งรอบต่อ profile ตัวเลขนี้ไม่ใช่
five-round benchmark จึงรายงานเป็นค่ารอบเดียว

| Profile | Load s | Generation s | Pipeline s | Generation img/h | Pipeline img/h | Amortized gen p50/p95 | Peak alloc/reserved GiB | Peak RSS GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2B BF16 | 22.116 | 148.219 | 173.187 | 14,573.1 | 12,472.0 | 0.2578 / 0.2954 | 4.04 / 4.08 | 4.69 |
| 4B BF16 | 17.818 | 185.702 | 205.088 | 11,631.5 | 10,532.0 | 0.3298 / 0.3786 | 8.36 / 8.40 | 8.99 |
| 8B NF4 | 69.111 | 256.426 | 271.030 | 8,423.5 | 7,969.6 | 0.4679 / 0.5374 | 6.15 / 6.22 | 17.13 |
| 8B BF16 | 70.551 | 168.196 | 183.084 | 12,842.2 | 11,797.8 | 0.2762 / 0.3790 | 16.43 / 16.46 | 17.04 |

## 4. Experiment C1 — Batch scaling บน A100 หนึ่งใบ

ใช้ `4b-bf16`, 1,000 images, 5 รอบต่อ batch ไม่มี OOM และทุก prediction มี
status `ok`

| Batch | Load s | Generation s | Pipeline s | Generation img/h | Pipeline img/h | Amortized gen p50 | p95 | Peak alloc GiB | Peak reserved GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.254 | 305.282 | 330.451 | 11,792.4 | 10,894.2 | 0.3297 | 0.3711 | 8.36 | 8.39 |
| 2 | 5.349 | 182.314 | 205.996 | 19,746.1 | 17,476.0 | 0.1884 | 0.2112 | 8.46 | 8.64 |
| 4 | 5.290 | 106.116 | 130.715 | 33,925.3 | 27,540.8 | 0.1049 | 0.1161 | 8.61 | 8.91 |
| 8 | 5.309 | 63.410 | 86.509 | 56,773.7 | **41,614.4** | 0.0633 | 0.0674 | 8.92 | 9.62 |

Batch-level latency เพิ่มตามขนาด batch ขณะที่ amortized latency ลดลง:

| Batch | Batch generation p50 | p95 | Batch pipeline p50 | p95 | Allocated GPU-h/1,000 images |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.3297 | 0.3711 | 0.3499 | 0.4003 | 0.0995 |
| 2 | 0.3769 | 0.4224 | 0.4180 | 0.4781 | 0.0640 |
| 4 | 0.4197 | 0.4643 | 0.5263 | 0.5968 | 0.0417 |
| 8 | 0.5067 | 0.5395 | 0.6931 | 0.7934 | **0.0290** |

Peak process RSS ของ batch 1/2/4/8 เท่ากับ 8.99/9.00/8.99/8.99 GiB
ตามลำดับ

กฎ highest valid median pipeline throughput เลือก batch 8 สำหรับ C2

## 5. Experiment C2 — Multi-GPU strong scaling

ใช้ `4b-bf16`, per-GPU batch 8 และ workload คงที่ 2,000 ภาพต่อรอบ รวม 5 รอบ
ต่อจำนวน GPU แต่ละ rank ได้ shard ไม่ซ้ำและ CPU affinity 16 cores/rank

| GPUs | Median generation s | Median pipeline s | Median complete-round wall s | Aggregate img/h | Speedup | Efficiency | Peak alloc/rank GiB | Peak reserved/rank GiB | Peak RSS/rank GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 129.540 | 176.549 | 192.296 | 40,781.8 | 1.000 | 100.00% | 8.93 | 9.71 | 9.00 |
| 2 | 65.584 | 88.691 | 105.495 | 81,180.7 | 1.991 | 99.53% | 8.93 | 9.56 | 8.99 |
| 4 | 33.517 | 45.105 | 64.905 | 159,628.9 | 3.914 | 97.86% | 8.93 | 9.77 | 8.99 |

| GPUs | Scaling gate | ผล | Allocated GPU-h/1,000 images |
|---:|---:|---|---:|
| 1 | baseline | — | 0.0267 |
| 2 | efficiency ≥80% | ผ่าน | 0.0296 |
| 4 | efficiency ≥70% | ผ่าน | 0.0366 |

การเพิ่ม GPU ลด turnaround แต่ใช้ GPU-hours ต่อ 1,000 ภาพมากขึ้นเล็กน้อย
ผล unpinned และ CPU-pinned รุ่นแรกถูกเก็บเป็น diagnostic runs ไม่ถูกใช้แทนผล final
เพราะมี CPU contention/inter-op-thread confound


## ขอบเขตของผล

- ผลเป็น Development/Pilot จาก official Mendeley V3 train split; ไม่มี confirmatory Test metric
- การเทียบ Set A เป็นการเทียบแพลตฟอร์ม เพราะ Python/PyTorch/CUDA ต่างเวอร์ชัน
- ยังไม่มี output tokens/sec ที่สรุปเป็นผล, end-to-end turnaround หรือ CI ของ C2 speedup/efficiency
