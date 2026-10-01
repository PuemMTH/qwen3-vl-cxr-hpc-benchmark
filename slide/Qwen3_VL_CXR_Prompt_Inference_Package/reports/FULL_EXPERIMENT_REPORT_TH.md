# รายงานฉบับสมบูรณ์: การประเมิน Qwen3-VL สำหรับ Chest X-ray บน RTX 5070 และ LANTA A100

**วันที่ทดลอง:** 23 กันยายน 2026  
**สถานะการศึกษา:** Development/Pilot (`development_pilot_only`)  
**ขอบเขตข้อมูล:** ใช้เฉพาะข้อมูลจาก official V3 train split; ไม่เปิด confirmatory Test  
**หน่วยวิเคราะห์หลักด้านคุณภาพ:** ผู้ป่วย  
**หน่วยวิเคราะห์รอง:** ภาพ  

## บทสรุป

การทดลองแยกคำถามออกเป็นสี่ส่วน: ความเร็วต่อ GPU (A), คุณภาพและผลของ precision
(B), batch scaling (C1) และ multi-GPU scaling (C2) จากนั้นเพิ่ม Set D เพื่อทดสอบ
30B MoE และ 32B dense ซึ่งไม่สามารถโหลดบน RTX 5070 ได้

ผลหลักมีดังนี้

1. ที่ batch 1 การใช้ A100 หนึ่งใบไม่ทำให้โมเดล 2B–8B เร็วขึ้น RTX 5070 ให้
   pipeline throughput สูงกว่าในทุก profile โดย A100 speedup เท่ากับ 0.652,
   0.703 และ 0.786 เท่า จึงไม่ผ่านเกณฑ์ 1.5 เท่าทั้งหมด
2. กฎเลือก comparator ที่กำหนดไว้บน Development เลือก `4b-bf16` ด้วย
   patient-level balanced accuracy 0.6867
3. บน A100 การเพิ่ม batch จาก 1 เป็น 8 เพิ่ม pipeline throughput จาก 10,894
   เป็น 41,614 images/hour โดยไม่เกิด OOM หรือ parse failure
4. เมื่อแบ่งข้อมูลและ CPU affinity ถูกต้อง C2 ให้ speedup 1.991 เท่าบน 2 GPUs
   และ 3.914 เท่าบน 4 GPUs คิดเป็น scaling efficiency 99.53% และ 97.86%
5. 30B และ 32B เกิด CUDA OOM จริงบน RTX 5070 แต่รันสำเร็จบน 2×A100
   บน Development 549 ผู้ป่วย ทั้งสองโมเดลมี balanced accuracy สูงกว่า 4B
   แต่เพิ่มเพียง 0.0404 และ 0.0468 จึงไม่ถึง margin +0.05 ที่กำหนดไว้
   นอกจากนี้ 32B ทำให้ PNEUMONIA recall ลดลง 0.1086 ซึ่งเกินขอบเขต −0.03
6. งานทั้งหมดใช้ 9.3200 allocated GPU-hours จาก planning ceiling 24 GPU-hours

ข้อสรุปที่ข้อมูลรองรับคือ LANTA มีประโยชน์ด้าน **capacity, batching และ
parallelism** แต่ไม่ให้ความเร็วต่อ GPU สำหรับ workload batch 1 ที่ทดสอบ และยังไม่มี
large model ใดผ่าน quality gate ที่กำหนดไว้ งานนี้ไม่ใช่หลักฐานเชิงคลินิกและไม่ใช่
ผล confirmatory

## 1. ข้อมูล โมเดล และวิธีทดลอง

### 1.1 ชุดข้อมูลที่ใช้จริง

ต้นทางคือ Kermany, Zhang, and Goldbaum Mendeley Data V3 ซึ่ง canonical manifest
มี 5,856 ภาพจาก 3,081 ผู้ป่วย การทดลองทั้งหมดในรายงานนี้ดึงจาก official train
split และแยก labels ออกจาก blind inference manifest

| ชุดทดลอง | ภาพ | ผู้ป่วย | NORMAL | PNEUMONIA | การใช้งาน |
|---|---:|---:|---:|---:|---|
| Performance-1000 | 1,000 | 859 | 500 | 500 | A และ C1 |
| Quality-Development-600 | 600 | 549 | 300 | 300 | B และ D |
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
| `30b-a3b-bf16` | `Qwen/Qwen3-VL-30B-A3B-Instruct` | `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c` | BF16 | MoE |
| `32b-bf16` | `Qwen/Qwen3-VL-32B-Instruct` | `0cfaf48183f594c314753d30a4c4974bc75f3ccb` | BF16 | dense |

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

## 6. Set D — 30B/32B บน Development 600 ภาพ

Set D เป็น exploratory Development extension เพราะ 30B/32B ไม่อยู่ใน candidate
list เดิมของ Experiment B แต่ใช้ manifest และ comparator 4B ชุดเดียวกันเพื่อให้
paired comparison ตรวจสอบได้ ทั้งสองโมเดลรัน 2 รอบและ prediction hash ตรงกันข้ามรอบ

### 6.1 Patient-level metrics

| Profile | n | Accuracy | Balanced accuracy | Macro-F1 | Uncertainty | Coverage | Image BA |
|---|---:|---:|---:|---:|---:|---:|---:|
| 30B MoE BF16 | 549 | 0.7231 [0.6867, 0.7596] | 0.7271 [0.6909, 0.7628] | 0.7279 [0.6908, 0.7649] | 0.0255 | 0.9745 | 0.7400 |
| 32B dense BF16 | 549 | 0.7322 [0.6958, 0.7705] | 0.7335 [0.6968, 0.7711] | 0.7436 [0.7071, 0.7805] | 0.0310 | 0.9690 | 0.7533 |

| Profile | Class | Support | Predicted | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| 30B MoE | NORMAL | 282 | 193 | 0.8497 | 0.5816 | 0.6905 |
| 30B MoE | PNEUMONIA | 267 | 342 | 0.6813 | 0.8727 | 0.7652 |
| 32B dense | NORMAL | 282 | 243 | 0.7984 | 0.6879 | 0.7390 |
| 32B dense | PNEUMONIA | 267 | 289 | 0.7197 | 0.7790 | 0.7482 |

| Profile | NORMAL recall CI | PNEUMONIA recall CI | Uncertainty CI | Coverage CI |
|---|---:|---:|---:|---:|
| 30B MoE | [0.5213, 0.6384] | [0.8277, 0.9101] | [0.0128, 0.0401] | [0.9599, 0.9872] |
| 32B dense | [0.6348, 0.7447] | [0.7302, 0.8277] | [0.0182, 0.0455] | [0.9545, 0.9818] |

### 6.2 Paired delta เทียบ 4B BF16

| Candidate | Metric | Delta | 95% paired-bootstrap CI |
|---|---|---:|---:|
| 30B MoE | Accuracy | +0.0419 | [+0.0036, +0.0820] |
| 30B MoE | Balanced accuracy | +0.0404 | [+0.0025, +0.0789] |
| 30B MoE | Macro-F1 | +0.0478 | [+0.0077, +0.0902] |
| 30B MoE | Uncertainty rate | −0.0036 | [−0.0164, +0.0091] |
| 30B MoE | Coverage | +0.0036 | [−0.0091, +0.0164] |
| 30B MoE | Recall NORMAL | +0.0957 | [+0.0355, +0.1596] |
| 30B MoE | Recall PNEUMONIA | −0.0150 | [−0.0599, +0.0300] |
| 32B dense | Accuracy | +0.0510 | [+0.0073, +0.0984] |
| 32B dense | Balanced accuracy | +0.0468 | [+0.0043, +0.0937] |
| 32B dense | Macro-F1 | +0.0636 | [+0.0188, +0.1129] |
| 32B dense | Uncertainty rate | +0.0018 | [−0.0128, +0.0182] |
| 32B dense | Coverage | −0.0018 | [−0.0182, +0.0128] |
| 32B dense | Recall NORMAL | +0.2021 | [+0.1277, +0.2766] |
| 32B dense | Recall PNEUMONIA | −0.1086 | [−0.1610, −0.0524] |

| Candidate | Correct candidate/wrong comparator | Wrong candidate/correct comparator | Discordant | McNemar p |
|---|---:|---:|---:|---:|
| 30B MoE | 71 | 48 | 119 | 0.0433 |
| 32B dense | 102 | 74 | 176 | 0.0415 |

30B ไม่ผ่านเกณฑ์ Δ balanced accuracy ≥0.05 แม้ recall margin และ nominal McNemar
ผ่าน ส่วน 32B ไม่ผ่านทั้ง Δ≥0.05 และเกณฑ์ class-recall drop ≥−0.03 การทดสอบ
candidate สองตัวกับ comparator เดียวกันยังไม่มี multiplicity rule ที่ freeze ใน SAP;
หากใช้ Bonferroni α=0.025 ค่า p ทั้งสองจะไม่ผ่าน

### 6.3 Performance ของ large models

| Profile | Round | Load s | Generation s | Pipeline s | Generation img/h | Pipeline img/h | Batch generation p50/p95 | Batch pipeline p50/p95 | Peak alloc GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30B MoE | 1 | 59.137 | 479.996 | 496.478 | 4,500.0 | 4,350.6 | 0.8606 / 0.9521 | 0.8828 / 0.9767 | 58.72 | 60.19 |
| 30B MoE | 2 | 33.644 | 464.806 | 480.564 | 4,647.1 | 4,494.7 | 0.8365 / 0.9239 | 0.8555 / 0.9465 | 58.72 | 60.19 |
| 32B dense | 1 | 68.282 | 405.552 | 430.165 | 5,326.1 | 5,021.3 | 0.7663 / 0.8552 | 0.7917 / 0.8905 | 62.36 | 62.45 |
| 32B dense | 2 | 35.619 | 406.132 | 420.998 | 5,318.5 | 5,130.7 | 0.7683 / 0.8611 | 0.7831 / 0.8834 | 62.36 | 62.45 |

Peak process RSS อยู่ที่ประมาณ 58.72 GiB สำหรับ 30B และ 62.96 GiB สำหรับ 32B
ต่อ process โดยน้ำหนักถูกกระจายข้าม A100 สองใบ รอบแรกมี cold-cache/load overhead
สูงกว่า แต่ generation time ของ 32B แทบไม่เปลี่ยนระหว่างรอบ

## 7. Local capacity ceiling

การทดสอบบังคับ BF16 weights ทั้งหมดลง CUDA device 0 โดยไม่ใช้ CPU/disk offload
และตรวจว่าไฟล์น้ำหนักครบตาม registry ก่อนโหลด

| Profile | Weight files | Weight bytes | Standard loader | No-warm-up diagnostic | Peak allocated | Peak reserved |
|---|---:|---:|---|---|---:|---:|
| 30B MoE | 13 | 62,141,618,016 | OOM ขอ 10.30 GiB | OOM ขอเพิ่ม 768 MiB | 10,410,336,256 B | 10,550,771,712 B |
| 32B dense | 14 | 66,714,912,872 | OOM ขอ 10.30 GiB | OOM ขอเพิ่ม 80 MiB | 10,843,056,128 B | 10,863,247,360 B |

Standard-loader OOM เกิดที่ speculative allocator warm-up จึงใช้เป็นหลักฐานรอง
ส่วน no-warm-up run โหลด weights จริงจนเหลือหน่วยความจำเพียงเล็กน้อยแล้ว OOM
เป็นหลักฐานหลักว่า local card ไม่สามารถรองรับ BF16 model ทั้งสองตัว

## 8. Secondary image-level metrics และ confusion matrices

### 8.1 Image-level summary

| Profile | n | Accuracy | Balanced accuracy | Macro-F1 | Uncertainty | Coverage | Status ok | Parse failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2B BF16 | 600 | 0.6733 | 0.6733 | 0.6470 | 0.0000 | 1.0000 | 600 | 0 |
| 4B BF16 | 600 | 0.7000 | 0.7000 | 0.6884 | 0.0000 | 1.0000 | 600 | 0 |
| 8B NF4 | 600 | 0.5967 | 0.5967 | 0.5941 | 0.0117 | 0.9883 | 600 | 0 |
| 8B BF16 | 600 | 0.4417 | 0.4417 | 0.5216 | 0.3267 | 0.6733 | 600 | 0 |
| 30B MoE | 600 | 0.7400 | 0.7400 | 0.7352 | 0.0017 | 0.9983 | 600 | 0 |
| 32B dense | 600 | 0.7533 | 0.7533 | 0.7528 | 0.0000 | 1.0000 | 600 | 0 |

`status ok` หมายถึง pipeline และ parser ทำงานครบ ไม่ได้หมายความว่าคำตอบไม่เป็น
`UNCERTAIN`

### 8.2 Image-level per-class metrics

| Profile | Class | Support | Predicted | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| 2B BF16 | NORMAL | 300 | 136 | 0.8824 | 0.4000 | 0.5505 |
| 2B BF16 | PNEUMONIA | 300 | 464 | 0.6121 | 0.9467 | 0.7435 |
| 4B BF16 | NORMAL | 300 | 184 | 0.8261 | 0.5067 | 0.6281 |
| 4B BF16 | PNEUMONIA | 300 | 416 | 0.6442 | 0.8933 | 0.7486 |
| 8B NF4 | NORMAL | 300 | 219 | 0.6484 | 0.4733 | 0.5472 |
| 8B NF4 | PNEUMONIA | 300 | 374 | 0.5775 | 0.7200 | 0.6409 |
| 8B BF16 | NORMAL | 300 | 246 | 0.6585 | 0.5400 | 0.5934 |
| 8B BF16 | PNEUMONIA | 300 | 158 | 0.6519 | 0.3433 | 0.4498 |
| 30B MoE | NORMAL | 300 | 213 | 0.8404 | 0.5967 | 0.6979 |
| 30B MoE | PNEUMONIA | 300 | 386 | 0.6865 | 0.8833 | 0.7726 |
| 32B dense | NORMAL | 300 | 272 | 0.7794 | 0.7067 | 0.7413 |
| 32B dense | PNEUMONIA | 300 | 328 | 0.7317 | 0.8000 | 0.7643 |

### 8.3 Confusion matrices

คอลัมน์เรียงเป็น true NORMAL → predicted NORMAL/PNEUMONIA/UNCERTAIN และ true
PNEUMONIA → predicted NORMAL/PNEUMONIA/UNCERTAIN

| Profile | Level | N→N | N→P | N→U | P→N | P→P | P→U |
|---|---|---:|---:|---:|---:|---:|---:|
| 2B BF16 | patient | 110 | 165 | 7 | 13 | 251 | 3 |
| 2B BF16 | image | 120 | 180 | 0 | 16 | 284 | 0 |
| 4B BF16 | patient | 137 | 134 | 11 | 25 | 237 | 5 |
| 4B BF16 | image | 152 | 148 | 0 | 32 | 268 | 0 |
| 8B NF4 | patient | 128 | 146 | 8 | 64 | 187 | 16 |
| 8B NF4 | image | 142 | 158 | 0 | 77 | 216 | 7 |
| 8B BF16 | patient | 146 | 48 | 88 | 70 | 85 | 112 |
| 8B BF16 | image | 162 | 55 | 83 | 84 | 103 | 113 |
| 30B MoE | patient | 164 | 109 | 9 | 29 | 233 | 5 |
| 30B MoE | image | 179 | 121 | 0 | 34 | 265 | 1 |
| 32B dense | patient | 194 | 81 | 7 | 49 | 208 | 10 |
| 32B dense | image | 212 | 88 | 0 | 60 | 240 | 0 |

## 9. Resource use และ operational metrics

| กลุ่มงาน | Job IDs | GPU-hours |
|---|---|---:|
| Large-model feasibility 20 images | 6293741–6293742 | 0.2144 |
| Experiment A | 6293861–6293863 | 1.6119 |
| Experiment B | 6293864–6293867 | 0.3319 |
| Original C1 diagnostic | 6293868–6293871 | 1.2081 |
| Selected 4B C1 | 6294007–6294010 | 1.1711 |
| C2 unpinned diagnostic | 6294156–6294158 | 1.5625 |
| C2 CPU-pinned diagnostic | 6294184–6294186 | 1.0075 |
| C2 final CPU-pinned v2 | 6294203–6294205 | 0.9286 |
| Set D 30B/32B | 6295763–6295764 | 1.2839 |
| **รวม** |  | **9.3200** |

Selected job queue waits และ allocation elapsed:

| งาน | GPUs | Queue wait | Allocation elapsed | Result |
|---|---:|---:|---:|---|
| A 2B / 4B / 8B | 1 | 5 s | 25:26 / 30:37 / 40:40 | complete |
| C1 4B batch 1/2/4/8 | 1 | 00:30 / 06:54 / 10:25 / 14:28 | 29:51 / 19:12 / 12:31 / 08:42 | complete |
| C2 final 1/2/4 GPU | 1/2/4 | 2 s | 16:01 / 08:53 / 05:29 | complete |
| Set D 30B / 32B | 2 | 30 s | 20:13 / 18:18 | complete |

Quota snapshot ล่าสุดก่อน Set D เหลือ 4,366.41/115,056.00 shared SHr (3.80%)
หน่วย SHr ไม่ถูกแปลงเป็น guaranteed A100 GPU-hours เพราะไม่มี conversion policy
ที่ยืนยันแล้ว

## 10. การตรวจความครบถ้วนและ reproducibility

- State files 25 ชุดมีสถานะ `complete`
- Automated tests ผ่าน 9/9
- C2 ตรวจครบ 15 รอบว่าแต่ละ image ID ปรากฏครั้งเดียว ไม่มี shard ซ้ำหรือหาย
- Set D evaluation checksum ผ่านทั้ง 30B และ 32B
- Set D prediction hashes ตรงกันระหว่างรอบ 1 และ 2 ทั้งสองโมเดล
- Execution pack ตรวจผ่าน 73/73 ไฟล์; manifest SHA-256:
  `bced27c6fe894d3254aaf2fd1885d384d7eb8e628ef9358411bf79fd3aa575c4`
- Evidence manifest ครอบคลุม 1,199 ไฟล์; SHA-256:
  `bce7df5296a584f1445d2ecae9498b1aa2ec44287fb0a0b5e44fcdfb9f12a9c9`
- รายงานฉบับนี้สร้างหลัง evidence manifest จึงไม่ใส่ตัวเองใน manifest เพื่อหลีกเลี่ยง
  self-reference; checksum ของรายงานอยู่ใน `FULL_EXPERIMENT_REPORT_TH.md.sha256`

มี code drift ระหว่าง Experiment A: `inference.py` เพิ่ม CPU/thread telemetry และ
จำกัด local threads ระหว่าง local 4B round 1 กับ round 2–5 การเปรียบเทียบธรรมชาติ
ใน 4B พบผลต่างประมาณ 0.3% ซึ่งไม่เปลี่ยนทิศทางของ speed result แต่ยังต้องถือเป็น
ข้อจำกัดด้าน provenance และต้อง freeze pack ใหม่ก่อน confirmatory

## 11. Metric ที่ยังไม่มีหลักฐาน

- ไม่มี confirmatory Test metric เพราะ Test ยังล็อกอยู่
- ไม่มี single complete-turnaround timestamp ตั้งแต่เริ่ม stage model จนเก็บผลกลับ
  local; มีเฉพาะ queue, allocation elapsed, per-round wall และ model-loading time
- ไม่มี energy consumption, power efficiency หรือค่าใช้จ่ายทางการเงิน
- C2 ไม่มี confidence interval ของ speedup/efficiency
- ไม่มี post-Set-D quota snapshot
- ยังไม่มี multiplicity procedure ที่นักสถิติอนุมัติสำหรับ large candidates สองตัว
- ไม่มี human HPC operator sign-off, frozen SAP, IRB/privacy determination หรือ
  confirmatory access authorization

## 12. ข้อสรุปและคำตัดสินตามเกณฑ์ที่กำหนด

| Gate | เกณฑ์ | ผล | คำตัดสิน |
|---|---|---|---|
| Hardware speed | A100 speedup ≥1.5 และ lower CI >1 | 0.652–0.786 | ไม่ผ่าน |
| 8B BF16 quality | Δ BA ≥+0.05, p<0.05, recall drop ≥−0.03 | Δ BA −0.2687 vs 4B; recall PNEUMONIA −0.5693 | ไม่ผ่าน |
| 30B quality | Δ BA ≥+0.05, p<0.05, recall drop ≥−0.03 | +0.0404; p=0.0433; min recall delta −0.0150 | ไม่ผ่าน margin |
| 32B quality | เงื่อนไขเดียวกัน | +0.0468; p=0.0415; PNEUMONIA −0.1086 | ไม่ผ่าน margin และ recall |
| Batch scaling | throughput สูงสุดโดยไม่ OOM/failure | batch 8, 41,614 img/h | เลือก batch 8 |
| 2-GPU scaling | efficiency ≥80% | 99.53% | ผ่าน |
| 4-GPU scaling | efficiency ≥70% | 97.86% | ผ่าน |
| Budget | ≤24 GPU-hours | 9.3200 | ผ่าน |

สำหรับ workload batch 1 และโมเดลที่พอดีกับ local GPU ควรใช้ RTX 5070 หากเป้าหมาย
คือเวลา inference ต่อ GPU ส่วน LANTA เหมาะเมื่อจำเป็นต้องใช้ 30B/32B, batch ใหญ่
หรือ parallel throughput หลาย GPU ผล Development แสดงว่า large models เพิ่มคุณภาพ
บางส่วน แต่ยังไม่ถึงเกณฑ์ที่ตั้งไว้ จึงไม่ควรเปลี่ยน comparator หรือผ่อน threshold
ย้อนหลังจากผลนี้

Confirmatory analysis ต้องรอการปิด BLOCK-01–04, frozen SAP, touched-test history,
multiplicity rule และสิทธิ์เปิด labels ก่อน รายงานนี้ไม่ปลด lock ดังกล่าว

## 13. แหล่งหลักฐาน

- `systems-summary.json` และ `systems-summary.csv`: Experiment A/C1/C2 aggregate
- `lanta/B-development/*/evaluation/metrics.json`: Experiment B metrics
- `lanta/B-development/quality-summary.json`: selection rule และ 8B comparisons
- `lanta/C1-selected-4b-bf16/selection.json`: batch selection
- `lanta/C2-selected-4b-bf16-cpu-pinned-v2/`: final scaling states/logs
- `lanta/D-large-development/*/evaluation/metrics.json`: Set D quality/comparison
- `artifacts/local-capacity-30b*.json`, `artifacts/local-capacity-32b*.json`:
  local OOM evidence
- `slurm-accounting.tsv`: queue/start/end/exit/resources/MaxRSS
- `experiment-evidence.sha256`: file-level evidence integrity
