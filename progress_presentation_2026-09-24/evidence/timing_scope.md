# ขอบเขตเวลา inference pipeline

สำหรับตารางเวลา Experiment A/C1/C2 หน่วยเวลาคือ **วินาที (`s`)**

1. `pipeline_seconds` ต่อ batch เริ่มจับเวลาก่อนเปิดภาพจากดิสก์
2. ระหว่างจับเวลา: อ่านและปรับภาพ (รวม EXIF, RGB, ย่อขนาด) → จัดภาพและ prompt เป็น tensor แล้วส่งเข้า GPU → `model.generate` → decode token และ parse คำตอบ
3. `pipeline_seconds_total` ต่อการรันหนึ่ง rank = ผลรวม `pipeline_seconds` ของ batch ที่ `status=success` และ `warmup=false`
4. สำหรับ C2 เวลา inference pipeline ของหนึ่งรอบ = **ค่าสูงสุด**ของ `pipeline_seconds_total` ในทุก rank; `Median inference pipeline [s]` = **มัธยฐานของเวลารอบ 5 รอบ**

สำหรับ Experiment A (1 GPU, batch 1) ใช้ `pipeline_seconds_total` ของ RTX 5070 และ A100 ใน **รอบเดียวกัน** หารกัน แล้วหามัธยฐานของอัตราส่วน 5 รอบ:

$$
S_A=\operatorname{median}_{r=1,\ldots,5}\left(\frac{T_{\mathrm{RTX},r}\;[\mathrm{s}]}{T_{\mathrm{A100},r}\;[\mathrm{s}]}\right),\qquad
T_{p,r}=\sum_{b\in\text{successful, non-warm-up batches of platform }p\text{ in round }r}t_{b,p,r}.
$$

ตัวอย่าง 4B BF16 รอบ 3: RTX 5070 มี `generation_seconds_total 212.803 s` และส่วนอื่นใน pipeline รวม `18.778 s` จึงได้ `pipeline_seconds_total 231.581 s`; A100 มี `304.437 + 24.782 = 329.219 s`. อัตราส่วนรอบนี้ `231.581 / 329.219 = 0.7034×` เป็นค่ากลางของอัตราส่วน 5 รอบ ค่า “ส่วนอื่น” ได้จาก pipeline ลบ generation; **ไม่มีเวลารายขั้น**สำหรับอ่านภาพ, เตรียม tensor หรือ decode/parse

$$
t_{r,k}=\max_{j=1,\ldots,k}\left(\sum_{b\in\text{successful, non-warm-up batches of rank }j\text{ in round }r} t_{b,r,j}\right),
\qquad
T_k=\operatorname{median}_{r=1,\ldots,5}(t_{r,k})\;[\mathrm{s}]
$$

**ไม่รวม:** เวลาโหลดโมเดล, warm-up, รอคิวงาน และการประเมิน quality หลัง inference. `Median inference pipeline` จึงไม่ใช่ complete-round wall time หรือเวลาตั้งแต่ส่งงานจนเสร็จ

## เวลาอื่นที่มีในไฟล์ผล

| ค่า | ขอบเขต | ที่มาในชุดนำเสนอ |
|---|---|---|
| `model_loading_seconds` | โหลด model และ processor ให้พร้อมใช้งานต่อการรันหนึ่งรอบ | `evidence/runs/.../state.json` → `rounds[].performance` หรือ `rounds[].ranks[].performance` |
| `generation_seconds` | เวลา `model.generate` ของหนึ่ง batch โดย synchronize GPU ก่อนและหลัง | อยู่ใน `batches.jsonl` ของงานต้นทาง; ในชุดนี้มีผลรวมและสถิติใน `performance` |
| `pipeline_seconds` | เวลา pipeline ทั้งหมดของหนึ่ง batch ตามขั้นตอนด้านบน | อยู่ใน `batches.jsonl` ของงานต้นทาง; ในชุดนี้มีผลรวมและสถิติใน `performance` |
| `elapsed_seconds` | เวลา subprocess ของหนึ่งรอบ A/B/C1 ตั้งแต่เริ่มรันจน process จบ รวมการเริ่ม process และขั้นตอนนอก inference | `evidence/runs/.../state.json` → `rounds[].elapsed_seconds` |
| `wall_seconds` | เวลา C2 หนึ่งรอบตั้งแต่ launcher เริ่ม rank จนทุก rank จบ รวมการเริ่ม process/โหลดโมเดล/เขียนผลภายในรอบ | `evidence/runs/lanta/C2-.../gpu*/state.json` → `rounds[].wall_seconds` |
| Queue wait / allocation elapsed | เวลารอ SLURM และระยะเวลาที่ได้รับจัดสรร GPU ตาม scheduler | [ตารางทรัพยากร](resource_abc.md) |

**ค่าที่คำนวณต่อจากเวลาที่วัด:** `generation_seconds_total` และ `pipeline_seconds_total` เป็นผลรวม batch ที่สำเร็จหลัง warm-up; p50/p95 ของ batch latency และ generation วินาทีต่อภาพเป็นสถิติจาก batch; throughput (ภาพ/ชั่วโมง), speedup, efficiency และ GPU-hours เป็นค่าคำนวณ ไม่ใช่ตัวจับเวลาเพิ่มเติม

ตัวอย่าง C2 final ที่ 1/2/4 GPU: median generation = **129.540 / 65.584 / 33.517 s**, median inference pipeline = **176.549 / 88.691 / 45.105 s**, และ median complete-round wall = **192.296 / 105.495 / 64.905 s** ตามลำดับ ตัวเลขทั้งสามชุดมีขอบเขตต่างกัน [ตารางผล §5](report_abc.md)

**ยังไม่มีเวลาแยก**สำหรับอ่านภาพ, เตรียม tensor, ส่งเข้า GPU, decode หรือ parse เป็นรายขั้นตอน; โค้ดจับเวลาเฉพาะ generation กับ pipeline ทั้งก้อน

ตรวจจากโค้ดต้นทาง: [ตัวจับเวลาและขั้นตอน batch](../../ai-for-research/module-06-vlm/vlm-cxr-benchmark/vlm_cxr/inference.py), [การรวม batch ที่สำเร็จ](../../ai-for-research/module-06-vlm/vlm-cxr-benchmark/vlm_cxr/inference.py), [การเลือก rank ที่ช้าที่สุดและมัธยฐาน](../../ai-for-research/module-06-vlm/vlm-cxr-benchmark/tools/summarize_systems_matrix.py), [นิยาม metric](../../ai-for-research/module-06-vlm/vlm-cxr-benchmark/METRICS.md)
