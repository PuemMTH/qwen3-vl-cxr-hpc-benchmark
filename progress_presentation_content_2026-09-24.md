# เนื้อหานำเสนอความก้าวหน้า: Qwen3-VL สำหรับ Chest X-ray

ตรวจไฟล์ใน workspace วันที่ 24 กันยายน 2026 เอกสารนี้ใช้เฉพาะเนื้อหาจาก [slide ครั้งก่อน](HPC-slides.md), [slide ความก้าวหน้าล่าสุด](Bento_Slides.bento.html), [รายงานที่วางใน `slide/`](slide/Qwen3_VL_CXR_Full_Experiment_Report_TH.html) และหลักฐานของ Experiment A, B, C1, C2 ตามที่วางแผนไว้ครั้งก่อน **ตัด Set D (30B/32B) และงานใหม่อื่นออก**

## ผลยังอยู่ไหม

**ยังอยู่** ผล Development/Pilot วันที่ 23 กันยายนอยู่ทั้งในรายงาน สรุปผล ไฟล์สถานะรายชุด และ metrics รายโมเดล:

| สิ่งที่ใช้ | ไฟล์หลัก | สถานะที่ตรวจ |
|---|---|---|
| Slide ครั้งก่อน | [`HPC-slides.md`](HPC-slides.md), [`assets/slides/`](assets/slides/) | 14 ภาพ slide และภาพ X-ray ต้นฉบับ 2 ภาพยังอยู่ |
| Slide ความก้าวหน้าล่าสุด | [`Bento_Slides.bento.html`](Bento_Slides.bento.html) | เปิดอ่านโครงสร้างได้ 19 หน้า; ส่วน Set D ที่อยู่ใน deck นี้ไม่ถูกใช้ในเนื้อหาฉบับนี้ |
| รายงานที่อยู่ใน `slide/` | [`Qwen3_VL_CXR_Full_Experiment_Report_TH.html`](slide/Qwen3_VL_CXR_Full_Experiment_Report_TH.html), [Markdown ใน package](slide/Qwen3_VL_CXR_Prompt_Inference_Package/reports/FULL_EXPERIMENT_REPORT_TH.md) | HTML และ Markdown ตรงกับสำเนาในโฟลเดอร์ผลทดลองแบบ SHA-256; checksum ของ package ผ่านทุกไฟล์ |
| ผลรวมและผลดิบ | [`systems-summary.json`](ai-for-research/module-06-vlm/vlm-cxr-benchmark/runs/pilot-matrix-20260923/systems-summary.json), [`quality-summary.json`](ai-for-research/module-06-vlm/vlm-cxr-benchmark/runs/pilot-matrix-20260923/lanta/B-development/quality-summary.json), [`runs/pilot-matrix-20260923/`](ai-for-research/module-06-vlm/vlm-cxr-benchmark/runs/pilot-matrix-20260923/) | State ของ A, B, C1 และ C2 รุ่น final ที่ตรวจมี `status: complete`; มี metrics และไฟล์ prediction รายรอบ |
| ข้อมูลชุดใหม่ | [`data/mendeley-v3-chest-xray/`](data/mendeley-v3-chest-xray/), [`manifests/`](ai-for-research/module-06-vlm/vlm-cxr-benchmark/runs/pilot-matrix-20260923/manifests/) | manifest ของ Performance 1,000 ภาพ, Quality 600 ภาพ/549 ผู้ป่วย, Scaling 2,000 ภาพยังอยู่ |

**ข้อจำกัดของผลเก่า:** ตาราง local pilot 20 ภาพใน slide ครั้งก่อนยังอยู่ที่หน้า 5 และ manifest ตัวอย่าง 20 ภาพยังอยู่ใน [`data/hf-vision-chest-xray-pneumonia/filtered/`](data/hf-vision-chest-xray-pneumonia/filtered/) แต่ยังไม่พบ raw run log ที่ใช้คำนวณค่า latency/throughput 3,765, 4,148, 5,126 images/hour ในตารางเดิม จึงใช้อ้างว่า “เคยรายงานไว้” ได้ แต่ไม่ควรใช้เป็นตัวเลขเปรียบเทียบหลักกับการทดลองล่าสุด

## จับคู่กับสิ่งที่นำเสนอครั้งก่อน

| ครั้งก่อน | ผลล่าสุดที่ตอบคำถามนั้น | หลักฐานและจุดที่ต้องพูดให้ตรง |
|---|---|---|
| หน้า 2–5: ปัญหา VRAM, ข้อมูล และ local pilot | เปลี่ยนจาก pilot 20 ภาพของ `hf-vision` test split มาเป็น Development จาก Mendeley V3 **train split**; local benchmark ใหม่ใช้ 1,000 ภาพ × 5 รอบ | [slide เก่า](HPC-slides.md), [report §1–2](slide/Qwen3_VL_CXR_Prompt_Inference_Package/reports/FULL_EXPERIMENT_REPORT_TH.md), [manifest](ai-for-research/module-06-vlm/vlm-cxr-benchmark/runs/pilot-matrix-20260923/manifests/performance-1000/summary.json). เป็นคนละ dataset/split/protocol จึงไม่เทียบตัวเลข pilot เก่ากับผลใหม่โดยตรง |
| หน้า 6–7: Need เรื่อง capacity | 8B BF16 รันครบ 600 ภาพบน A100; peak allocated 16.43 GiB; local 12 GB ไม่พอสำหรับ profile นี้ | [report §3.4](slide/Qwen3_VL_CXR_Prompt_Inference_Package/reports/FULL_EXPERIMENT_REPORT_TH.md), [8B state](ai-for-research/module-06-vlm/vlm-cxr-benchmark/runs/pilot-matrix-20260923/lanta/B-development/8b-bf16/state.json) |
| หน้า 8 และ 10: Experiment A / ความเร็ว GPU | ที่ batch 1, RTX 5070 เร็วกว่า A100 สำหรับ 2B BF16, 4B BF16, 8B NF4; A100 speedup = 0.652, 0.703, 0.786 เท่า | [report §2](slide/Qwen3_VL_CXR_Prompt_Inference_Package/reports/FULL_EXPERIMENT_REPORT_TH.md), [systems summary](ai-for-research/module-06-vlm/vlm-cxr-benchmark/runs/pilot-matrix-20260923/systems-summary.json). เทียบ **แพลตฟอร์ม** เพราะ Python/PyTorch/CUDA ต่างเวอร์ชัน; ไม่ใช่ผลแยกตัว GPU อย่างเดียว |
| หน้า 8 และ 11: Experiment B / quality | Development 600 ภาพ/549 ผู้ป่วยเลือก 4B BF16 เป็น comparator ด้วย patient-level balanced accuracy 0.6867; 8B BF16 ได้ 0.4180 และ uncertain 36.43% | [report §3](slide/Qwen3_VL_CXR_Prompt_Inference_Package/reports/FULL_EXPERIMENT_REPORT_TH.md), [quality summary](ai-for-research/module-06-vlm/vlm-cxr-benchmark/runs/pilot-matrix-20260923/lanta/B-development/quality-summary.json). `UNCERTAIN` นับอยู่ในตัวหาร; McNemar p ต่ำแต่ทิศทางผลของ 8B BF16 แย่ลง |
| หน้า 9 และ 12: Experiment C / batch | 4B BF16 บน A100 หนึ่งใบ: batch 1 → 8 เพิ่ม throughput 10,894 → 41,614 ภาพ/ชั่วโมง (3.82 เท่า), ไม่มี OOM/parse failure | [report §4](slide/Qwen3_VL_CXR_Prompt_Inference_Package/reports/FULL_EXPERIMENT_REPORT_TH.md), [C1 states](ai-for-research/module-06-vlm/vlm-cxr-benchmark/runs/pilot-matrix-20260923/lanta/C1-selected-4b-bf16/). เป็นการเทียบภายใน A100; RTX 5070 ในชุดหลักมีเพียง batch 1 |
| หน้า 9 และ 12: Experiment C / หลาย GPU | งานคงที่ 2,000 ภาพ, batch 8: 2 GPU ได้ speedup 1.991 เท่า/efficiency 99.53%; 4 GPU ได้ 3.914 เท่า/97.86% | [report §5](slide/Qwen3_VL_CXR_Prompt_Inference_Package/reports/FULL_EXPERIMENT_REPORT_TH.md), [C2 states รุ่น CPU-pinned-v2](ai-for-research/module-06-vlm/vlm-cxr-benchmark/runs/pilot-matrix-20260923/lanta/C2-selected-4b-bf16-cpu-pinned-v2/). ใช้ผลรุ่น final ที่แก้ CPU contention แล้ว; ยังไม่มี CI ของ scaling |
| หน้า 12: เวลา/ทรัพยากร | มี queue และ allocation time แยกตามงาน; **ไม่ใช้ยอดรวม 9.3200 GPU-hours** เพราะรวม Set D ที่ตัดออก | [report §9, §11](slide/Qwen3_VL_CXR_Prompt_Inference_Package/reports/FULL_EXPERIMENT_REPORT_TH.md), [`slurm-accounting.tsv`](ai-for-research/module-06-vlm/vlm-cxr-benchmark/runs/pilot-matrix-20260923/slurm-accounting.tsv). ยังไม่มีเวลา end-to-end ตั้งแต่ stage ถึงดึงผลกลับ local; ไม่อ้างว่าลดเวลารวมแล้ว |
| หน้า 13 ที่เคยซ่อน: เงื่อนไขที่จะเปลี่ยนข้อสรุป | hardware speed gate และ 8B BF16 quality gate ไม่ผ่าน; batch 8 และ 2/4-GPU scaling ผ่าน | [report §12](slide/Qwen3_VL_CXR_Prompt_Inference_Package/reports/FULL_EXPERIMENT_REPORT_TH.md). ผลไม่ผ่านเป็นผลการทดลองที่ควรรายงาน ไม่ใช่ตัดออกจากการนำเสนอ |

## เนื้อหาพูดแบบสั้น (ประมาณ 5–7 นาที)

1. **ทวนโจทย์:** “ครั้งก่อนเรานำเสนอแผนใช้ Qwen3-VL จำแนกภาพ Chest X-ray เป็น NORMAL หรือ PNEUMONIA และถามว่า A100 บน LANTA ช่วยเรื่องความจุ ความเร็ว และคุณภาพหรือไม่”
2. **อธิบายการเปลี่ยนข้อมูลก่อนขึ้นผล:** “pilot เดิมมีเพียง 20 ภาพจาก test split ของอีกชุดข้อมูล ผลรอบนี้จึงจัด Development ใหม่จาก official V3 train split และกัน confirmatory Test ไว้ การวิเคราะห์คุณภาพหลักเป็นระดับผู้ป่วย 549 ราย”
3. **Experiment A:** “ที่ batch 1 และ 1 GPU การย้าย 2B–8B ไป A100 ไม่เพิ่มความเร็ว; speedup เหลือ 0.65–0.79 เท่าเมื่อเทียบ RTX 5070 ดังนั้นข้อดีของ LANTA ในงานนี้ไม่ใช่ความเร็วต่อ GPU สำหรับโมเดลเล็ก”
4. **Experiment B:** “8B BF16 รันได้บน A100 แต่ไม่ได้ตอบแม่นขึ้น 4B BF16 เป็นตัวที่เลือกได้ดีที่สุดตามเกณฑ์ Development ด้วย balanced accuracy 0.6867 ขณะที่ 8B BF16 ได้ 0.4180 และตอบ UNCERTAIN ราว 36% ของผู้ป่วย”
5. **Experiment C:** “A100 ได้ประโยชน์เมื่อเพิ่ม batch: 4B BF16 เพิ่มจาก 10,894 เป็น 41,614 ภาพต่อชั่วโมงที่ batch 8 และเมื่อตรึง workload 2,000 ภาพแล้วเพิ่ม GPU เป็น 2/4 ใบ speedup เท่ากับ 1.99/3.91 เท่า”
6. **สรุปตาม Need:** “capacity และ throughput แบบ batch/หลาย GPU มีหลักฐานรองรับ ส่วน quality ที่ดีขึ้นตามเกณฑ์ที่กำหนดยังไม่พบ ผลทั้งหมดเป็น Development/Pilot ยังไม่ใช่ผลยืนยันหรือหลักฐานเชิงคลินิก”

## จุดที่ต้องไม่สื่อเกินผล

- ตัวเลข pilot เก่า 20 ภาพ, Set A 1,000 ภาพ, Set B 600 ภาพ/549 ผู้ป่วย และ Set C2 2,000 ภาพ เป็นคนละการวัด ใช้ metric และตัวหารให้ถูกชุด
- ความเร็ว A100 ใน Set A เป็นการเทียบแพลตฟอร์มที่ software stack ต่างกัน; ไม่สรุปสาเหตุว่า GPU รุ่นหนึ่งช้ากว่าโดยตัวมันเอง
- ยังไม่มี output tokens/sec ที่สรุปเป็นผล, ไม่มี end-to-end turnaround, ไม่มี CI ของ C2 และไม่มี confirmatory Test result
- หากใช้ deck `Bento_Slides.bento.html` ตรง ๆ ต้องตัดหน้า 14 (Set D) และลบเนื้อหา Set D ในหน้า 1–2, 4, 9, 15–18 ด้วย รวมถึงปรับยอด GPU-hours และตารางตัดสินใจที่รวม Set D
