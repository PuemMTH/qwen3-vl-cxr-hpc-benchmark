# ความถูกต้องในการจำแนกภาพ — ชุดทดลอง B

ชุด Development B มี **600 ภาพ** (NORMAL 300, PNEUMONIA 300) ตัวเลขด้านล่างดึงจาก `secondary_image_level` ในไฟล์ `evaluation/metrics.json` ของแต่ละแบบที่ทดสอบ โดยไม่รวมจำนวนผู้ป่วยในการนำเสนอ

| แบบที่ทดสอบ | ภาพ | Balanced accuracy | Accuracy | Macro-F1 | UNCERTAIN / ภาพ |
|---|---:|---:|---:|---:|---:|
| [2B BF16](runs/lanta/B-development/2b-bf16/evaluation/metrics.json) | 600 | 0.6733 | 0.6733 | 0.6470 | 0.00% |
| [4B BF16](runs/lanta/B-development/4b-bf16/evaluation/metrics.json) | 600 | **0.7000** | 0.7000 | 0.6884 | 0.00% |
| [8B NF4](runs/lanta/B-development/8b-nf4/evaluation/metrics.json) | 600 | 0.5967 | 0.5967 | 0.5941 | 1.17% |
| [8B BF16](runs/lanta/B-development/8b-bf16/evaluation/metrics.json) | 600 | 0.4417 | 0.4417 | 0.5216 | **32.67%** |

ทั้งสี่แบบทำนายภาพครบ 600/600; ค่า `UNCERTAIN` นับผิดในตัวหารของ metrics ข้างต้น ชุดข้อมูลสมดุลคลาส 300/300 จึงได้ accuracy เท่ากับ balanced accuracy ในผลนี้

**ขอบเขตการตีความ:** ตารางนี้บรรยายคำทำนายของภาพ 600 ภาพสำหรับสไลด์ความก้าวหน้า กฎเลือก comparator และ quality gate ในรายงานต้นทางใช้ metric หลักอีกระดับหนึ่ง จึงไม่อ้างว่าตารางนี้เป็นผลการตัดสินตาม gate เดิมหรือผล confirmatory Test

สไลด์ B ปัจจุบันใช้ [ตารางเทียบ RTX 5070 กับ A100 บนภาพเดียวกัน](quality_cross_platform_600.md) เป็นผลหลัก; ตารางหน้านี้เก็บผล A100 ทั้งสี่แบบไว้เป็นหลักฐานประกอบ รวม 8B BF16 ที่ไม่มีคู่เทียบ RTX
