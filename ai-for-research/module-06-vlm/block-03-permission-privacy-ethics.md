# BLOCK-03 — Permission, Privacy, License and Ethics

**Status:** `PARTIALLY RESOLVED — OWNER AND INSTITUTIONAL DETERMINATION OPEN`  
**Audit date:** 2026-09-23  
**Scope:** official Mendeley Data V3 CXR subset stored locally for this VLM study

## 1. Answers supported by public records

| Question | Verified answer |
|---|---|
| Dataset type | Public research dataset |
| Record/version | Kermany, Zhang, and Goldbaum, Mendeley Data V3; DOI `10.17632/rscbjbr9sj.3` |
| License | CC BY 4.0 |
| Permitted by license | copy, redistribute and adapt, including across media/formats and for any purpose, subject to license conditions |
| Required attribution | credit the creator, link the license and indicate changes |
| Important boundary | CC BY 4.0 does not itself clear privacy, publicity, moral, data-protection or institutional-policy obligations |
| Original collection ethics | the original paper reports institutional review/ethical controls for the source study; this does not constitute approval of this new experiment by the user's institution |
| Candidate model-weight licenses | official Hugging Face records for `Qwen/Qwen3-VL-{2B,4B,8B}-Instruct` state Apache-2.0; public and not gated at audit time |

Official sources:

- Mendeley Data V3: <https://data.mendeley.com/datasets/rscbjbr9sj/3>
- CC BY 4.0 legal code: <https://creativecommons.org/licenses/by/4.0/legalcode>
- Original Cell paper: <https://doi.org/10.1016/j.cell.2018.02.010>
- Qwen3-VL official records: <https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct>,
  <https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct>,
  <https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct>

Model-record revisions observed on 2026-09-23:

| Model | Revision | License |
|---|---|---|
| Qwen3-VL-2B-Instruct | `89644892e4d85e24eaac8bacfd4f463576704203` | Apache-2.0 |
| Qwen3-VL-4B-Instruct | `ebb281ec70b05090aa6165b016eac8ec08e71b17` | Apache-2.0 |
| Qwen3-VL-8B-Instruct | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | Apache-2.0 |

These revisions are license evidence and candidate reproducibility pins; the experiment
configuration still needs to freeze which revisions will actually be executed.

## 2. Local privacy and metadata evidence

The aggregate audit scanned all 5,856 V3 JPEG files:

- 1,586 files contain an EXIF container
- 217 contain `DateTime`, `Software`, and technical EXIF tags
- zero files contain the selected sensitive EXIF tag names checked by the audit
- 217 contain XMP payloads and 1,586 contain Photoshop metadata payloads whose values
  have not been interpreted for identifiers
- no pixel-level OCR/burned-in identifier review has been performed

Therefore the evidence supports “no selected sensitive EXIF tags found,” but it does
not support the stronger statement “the dataset is confirmed de-identified.”

Evidence:
`data/mendeley-v3-chest-xray/metadata_audit.json`, SHA-256
`56a9d5d96c7bdb3daefa57b2a3cb305af4efed7cc2a6c2eea4399f8c22724efd`.

## 3. Local access-control audit

Current filesystem state on 2026-09-23:

| Path | Mode | Consequence |
|---|---:|---|
| `/home/puem` | `750` | only owner and members of group `puem` can traverse the home directory |
| group `puem` | only member: `puem` | no separate unprivileged collaborator currently inherits group access |
| `data/mendeley-v3-chest-xray/` | `775` | subtree mode is broad, but the restrictive home-directory ancestor prevents access by unrelated local users |
| `data/mendeley-v3-chest-xray/manifest.csv` | `664` | file mode alone is readable by others, but they cannot traverse `/home/puem` under the observed account/group state |

No additional ACL entries were reported. Effective access for ordinary local accounts
is currently owner-only (`puem`); privileged administrators remain outside this Unix
mode boundary. Tightening the dataset subtree to `700/600` would add defense in depth,
but is not required to describe the currently observed effective access. HPC copies,
backups, future collaborators and export destinations still need their own access list.

## 4. Answers required from the owner

1. Confirm whether `puem` is the only permitted local user and name any permitted
   users/groups for future HPC or backup copies.
2. Approve or revise the proposed future access policy: owner-only by default;
   explicitly grant named collaborators when required.
3. Choose a retention trigger/date and deletion method for local, HPC, backup, cache
   and derived copies.
4. Specify the approved backup location and whether copies may leave the institution.
5. Confirm that the recorded model revisions are the revisions that will actually be
   executed; license identification itself is complete.

## 5. Answers required from institutional authority

1. Whether this secondary analysis is exempt, not-human-subjects research, or requires
   IRB/privacy/data-governance review.
2. Whether the available de-identification evidence is sufficient, including treatment
   of XMP/Photoshop metadata and possible burned-in pixel identifiers.
3. Whether the proposed local/HPC access, retention, backup and export plan complies
   with institutional policy.

The original study's ethics review and the public license are supporting evidence;
neither can substitute for a written determination required by the user's institution.

## 6. Closure rule

BLOCK-03 can be closed only after the owner decisions in section 4 are frozen and the
applicable institutional authority provides or explicitly waives the written
determination in section 5.
