# เปรียบเทียบความถูกต้องในการจำแนกภาพ: RTX 5070 กับ A100

## ที่มาของคู่เทียบ

ชุด B มี 600 ภาพ (NORMAL 300, PNEUMONIA 300) ทุก image ID และ SHA-256 อยู่ในชุด A จำนวน 1,000 ภาพครบ 600/600 จึงคัดผลทำนายภาพเดียวกันจากรอบที่ 1 ของ Experiment A บน RTX 5070 มาเทียบกับผล Experiment B บน A100 สำหรับ 2B BF16, 4B BF16 และ 8B NF4 **ไม่ได้มีการรัน B บน RTX 5070 โดยตรง** ผลทั้ง 5 รอบของ A บน RTX และ A100 ให้คำทำนายตรงกันภายในเครื่องทุกภาพที่อยู่ใน B; ผล A100 จาก A กับ B ตรงกันทั้ง 600 ภาพในสามแบบนี้

Config SHA-256 ของ Experiment A ตรงกันระหว่าง RTX และ A100 ในแต่ละแบบ (2B `05ccd90b…`, 4B `efd5880e…`, 8B NF4 `698e5802…`) และ model revision ในผลทำนายตรงกัน ภาพและ hash ตรงกันทุกภาพ จึงใช้เทียบผลรายภาพได้ Config ของ A100 B ต่างจาก A ในจำนวนภาพที่เลือกและจำนวน warm-up batch แต่คำทำนายบนภาพ 600 ภาพตรงกับ A100 A ทั้งหมด ส่วนการยืนยันว่าไฟล์โค้ดและ software stack เหมือนกันทุกส่วนยังต้องตรวจเพิ่มเติม

## ผลจากภาพ 600 ภาพ

`Balanced accuracy` คิดจากค่าเฉลี่ย recall ของ NORMAL กับ PNEUMONIA; `UNCERTAIN` ไม่ถูกนับเป็นคำตอบถูก ในชุดที่มี 300 ภาพต่อคลาสนี้ค่า balanced accuracy เท่ากับ accuracy รวม

| โมเดล | RTX 5070 image BA | A100 image BA | RTX UNCERTAIN | A100 UNCERTAIN | คำทำนายรายภาพต่างกัน |
|---|---:|---:|---:|---:|---:|
| 2B BF16 | 0.6683 (66.83%) | 0.6733 (67.33%) | 0/600 | 0/600 | **17/600** |
| 4B BF16 | 0.7017 (70.17%) | 0.7000 (70.00%) | 0/600 | 0/600 | **11/600** |
| 8B NF4 | 0.5950 (59.50%) | 0.5967 (59.67%) | 6/600 (1.00%) | 7/600 (1.17%) | **19/600** |

ภาพรวม: ค่าความถูกต้องใกล้กัน แต่คำทำนาย **ไม่เหมือนกันทุกภาพ** ในทั้งสามแบบ สาเหตุของภาพที่คำตอบต่างกัน **ยังไม่ได้พิสูจน์** และไม่ควรอธิบายจาก GPU หรือ software stack โดยการคาดเดา

8B BF16 เป็นกรณีที่รัน B ได้เฉพาะ A100: image BA `0.4417` และ UNCERTAIN `196/600 = 32.67%` จึงแสดงแยกจากกราฟคู่เทียบ RTX–A100

## หลักฐานต้นทาง

- ชุดภาพ: `ai-for-research/module-06-vlm/vlm-cxr-benchmark/runs/pilot-matrix-20260923/manifests/{performance-1000,quality-development-600}/blind_manifest.csv` และ `quality-development-600/labels.csv`
- RTX: `.../local/A/{2b-bf16,4b-bf16,8b-nf4}/round-01/intermediate/predictions.jsonl` (ตรวจทั้ง 5 รอบ)
- A100: `.../lanta/B-development/{2b-bf16,4b-bf16,8b-nf4}/evaluation/joined_predictions.csv`; เทียบกับ `.../lanta/A/.../predictions.jsonl` เพื่อยืนยันว่าผล A100 A กับ B ตรงกันบนภาพชุดนี้
- สถานะและ config hash: `evidence/runs/local/A/` และ `evidence/runs/lanta/B-development/` ในชุดนำเสนอนี้

ไฟล์ผลทำนายรายภาพอยู่ใน workspace ต้นทาง ไม่คัดลอกข้อมูลระดับภาพ/ผู้ป่วยเข้าชุดนำเสนอ
