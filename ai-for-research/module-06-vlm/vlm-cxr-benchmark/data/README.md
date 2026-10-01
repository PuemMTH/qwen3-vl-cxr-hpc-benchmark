# Data staging

Expected layout:

```text
data/dataset/
├── manifest.csv
└── images/{train,test}/{NORMAL,PNEUMONIA}/*.jpeg
```

Locally, `data/dataset` may be a symlink to
`/home/puem/Desktop/HPCP/data/mendeley-v3-chest-xray`. On LANTA, copy the canonical
dataset and verify the manifest/file hashes before running preprocess.

Do not replace `manifest.csv` with the legacy V2/Hugging Face manifest.
