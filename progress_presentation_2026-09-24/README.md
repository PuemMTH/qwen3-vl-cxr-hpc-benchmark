# ชุดข้อมูลสำหรับนำเสนอความก้าวหน้า Qwen3-VL CXR

ขอบเขต: Experiment A, B, C1, C2 และแผนจาก slide ครั้งก่อน

## Figma Slides ที่ปักไว้

**[เปิดตำแหน่ง slide progress ใน Figma](https://www.figma.com/slides/cYF2Km81QCHiRwhM1RavcL/HPC?node-id=164-54&t=L6sAjpPpDHp1OwC1-0)** · node `164:54`

**[ดาวน์โหลด PDF ชุด Progress 15 หน้า](HPC_Progress_A_B_C_2026-09-25_15slides.pdf)** — เพิ่ม Current Experimental Setup เป็นหน้า 3; เก็บ PDF 14 หน้าเดิมไว้แยกต่างหาก

หน้า 3 ระบุ code environment ของผล Set A จาก [รายงานผล](evidence/report_abc.md): RTX 5070 ใช้ Python 3.12.13 / PyTorch 2.14.0+cu130 ส่วน A100 ใช้ Python 3.11.10 / PyTorch 2.7.1+cu126 แม้ภาพ พรอมป์ model revision และ config hash จะตรงกัน การเทียบความเร็วจึงเป็นการเทียบแพลตฟอร์มทั้งชุด การรัน LANTA ด้วย cu130 ในภายหลังยังไม่มี local rerun ที่จับคู่ครบ จึงไม่ได้แทนตัวเลขในชุดนำเสนอนี้

อัปเดตวันที่ 25 กันยายน 2026: แถว **Progress Update** ในไฟล์ Figma มี 15 หน้า ใช้ผล A/B/C1/C2 และไม่รวม Set D เพิ่มหน้า Capacity หลังวิธีวัด เพื่อเชื่อมผล RTX 5070 ที่เกิด OOM ใน pilot เดิมกับผล 8B BF16 ที่รันสำเร็จบน A100 รอบนี้ โดยระบุว่าคนละชุดภาพและคนละรอบทดลอง

| หน้า Progress | เนื้อหา | Figma node |
|---:|---|---|
| 1 | หน้าปก | [164:54](https://www.figma.com/slides/cYF2Km81QCHiRwhM1RavcL/HPC?node-id=164-54) |
| 2 | คำถามจากครั้งก่อน | [164:71](https://www.figma.com/slides/cYF2Km81QCHiRwhM1RavcL/HPC?node-id=164-71) |
| 3 | Current Experimental Setup: task, models, hardware, evaluation, code environment | [253:54](https://www.figma.com/slides/cYF2Km81QCHiRwhM1RavcL/HPC?node-id=253-54) |
| 4 | ภาพอธิบาย batch 4 และเหตุผลที่เพิ่มภาพต่อครั้ง | [194:85](https://www.figma.com/slides/cYF2Km81QCHiRwhM1RavcL/HPC?node-id=194-85) |
| 5 | code box ของพรอมป์จริงจากไฟล์สร้าง config | [214:54](https://www.figma.com/slides/cYF2Km81QCHiRwhM1RavcL/HPC?node-id=214-54) |
| 6 | ข้อมูลและวิธีวัด | [173:69](https://www.figma.com/slides/cYF2Km81QCHiRwhM1RavcL/HPC?node-id=173-69) |
| 7 | ตารางเวลาที่วัดทั้งหมด: โค้ด 5 ชนิด และ SLURM 2 ชนิด | [202:54](https://www.figma.com/slides/cYF2Km81QCHiRwhM1RavcL/HPC?node-id=202-54) |
| 8 | ตาราง Capacity: 2B BF16, 4B BF16, 8B NF4 และ 8B BF16 บน RTX/A100 | [243:75](https://www.figma.com/slides/cYF2Km81QCHiRwhM1RavcL/HPC?node-id=243-75) |
| 9 | Experiment A และนิยามเวลาที่ใช้คิดอัตราส่วน A100 เทียบ RTX | [173:93](https://www.figma.com/slides/cYF2Km81QCHiRwhM1RavcL/HPC?node-id=173-93) |
| 10 | กราฟ A: RTX เทียบ A100 ที่ batch 1 | [194:54](https://www.figma.com/slides/cYF2Km81QCHiRwhM1RavcL/HPC?node-id=194-54) |
| 11 | กราฟ B: ความถูกต้อง RTX เทียบ A100 และคำถามเรื่องสาเหตุ | [174:54](https://www.figma.com/slides/cYF2Km81QCHiRwhM1RavcL/HPC?node-id=174-54) |
| 12 | Experiment C1/C2 พร้อมแท่งข้อมูลและนิยามเวลา | [174:84](https://www.figma.com/slides/cYF2Km81QCHiRwhM1RavcL/HPC?node-id=174-84) |
| 13 | กราฟ A100 batch 2 สูงกว่าค่า RTX 5070 batch 1 | [191:54](https://www.figma.com/slides/cYF2Km81QCHiRwhM1RavcL/HPC?node-id=191-54) |
| 14 | สรุปตามคำถามเดิม | [174:134](https://www.figma.com/slides/cYF2Km81QCHiRwhM1RavcL/HPC?node-id=174-134) |
| 15 | สิ่งที่จะทำต่อ | [174:156](https://www.figma.com/slides/cYF2Km81QCHiRwhM1RavcL/HPC?node-id=174-156) |

## เริ่มอ่าน

1. [แบบร่าง slide แยกทีละหน้า](slide_design_progress_abc.md) — ข้อความและแนวจัดวางของหน้า Progress ที่ใส่ใน Figma แล้ว
2. [ลำดับนำเสนอ 15 slide](presentation_flow_abc.md) — หัวข้อ ตัวเลขบนจอ และคำพูดสำหรับประมาณ 12 นาที
3. [เนื้อหาและบทพูด](presentation_content.md) — แผนเดิมเทียบผลจริง พร้อมสคริปต์ย่อ
4. [Slide ครั้งก่อน](HPC-slides.md) — ข้อความและภาพครบ 14 หน้า; รูปอยู่ใน `assets/`
5. [รายงานผลคัดเฉพาะ A/B/C](evidence/report_abc.md) — วิธีทดลองและตัวเลขหลัก
6. [ตารางเวลาและทรัพยากร A/B/C](evidence/resource_abc.md)
7. [ขอบเขตเวลา inference pipeline](evidence/timing_scope.md) — ขั้นตอนที่จับเวลาและสูตรมัธยฐานของ C2
8. [ค่าที่ใช้ทำกราฟจุดแซง](evidence/throughput_crossover.md) — ค่า RTX batch 1 เทียบ A100 batch 1/2/4/8 และข้อจำกัดของการเทียบ
9. [หลักฐานเทียบความถูกต้อง RTX–A100](evidence/quality_cross_platform_600.md) — ภาพเดียวกัน 600 ภาพ ผล balanced accuracy และจำนวนภาพที่ทำนายต่างกัน
10. [ผลตรวจความสอดคล้องของสไลด์](slide_consistency_audit_2026-09-25.md) — คำเรียก หน่วย สูตร และตัวเลขเทียบกับหลักฐานต้นทาง

## หลักฐานประกอบ

- `evidence/systems-summary.json`: ผลรวมความเร็วและ scaling
- [ความถูกต้องในการจำแนกภาพ](evidence/quality_image_level.md): ตารางผล A100 600 ภาพที่ใช้ประกอบ slide B
- `evidence/quality-summary.json`: หลักฐานกฎเลือก comparator ตามรายงานต้นทาง (คนละหน่วยวิเคราะห์กับตารางที่ใช้บน slide B)
- `evidence/runs/`: state ของ A/B/C1/C2 รุ่น final และ metrics ของ B
- `evidence/manifests/`: จำนวนภาพ/ผู้ป่วยและ provenance ของชุดทดลอง
- `evidence/configs/`: model revision ของ 2B/4B/8B และ config ที่ใช้ใน A/B/C
- `evidence/old-pilot/`: manifest 20 ภาพจากงานที่นำเสนอครั้งก่อน

ไฟล์นี้เป็นชุดสำหรับเตรียมการนำเสนอ จึงไม่มีภาพ CXR ทั้ง dataset, model weights หรือ prediction logs ขนาดใหญ่ หลักฐานต้นฉบับยังอยู่ใน workspace เดิม ผล local pilot ครั้งก่อนมีเฉพาะตารางใน slide และยังไม่พบ raw run log สำหรับตัวเลข throughput เดิม
