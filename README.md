# General
Repository associated with the article "In vivo autofluorescence lifetime imaging of the Drosophila brain reveals metabolic shifts associated with memory formation"

# Dataset
The data associated with this study are available in the [BioStudies database](https://www.ebi.ac.uk/biostudies) under accession number S-BIAD1528 (DOI: [10.6019/S-BIAD1528](https://doi.org/10.6019/S-BIAD1528)). Files should be downloaded through FTP or HTTP to preserve the directory structure (`/FLIM-marker`, `/markers` and `/file_lists`).

# Content
This repository contains a Python package (`mbflim`) and scripts reproducing the analyses of the article. Scripts are divided into mapping (using markers imaging data) and FLIM (using FLIM-marker imaging data and the output of mapping scripts).

# Installation
To be completed.

# Configuration
Change the paths in `config.json` so that 'source_dpath' points to the downloaded data, `output_dpath` to an output directory and `tmp_dpath` to a directory for temporary files storage.

# Usage
`S0_create_files_info.py` should be executed first to generate `files.csv` files in each subdirectory from the `.tsv` files in `/file_lists`.

Mapping scripts should be executed in this order:
- `M0_preprocess.py` with `study = 'MB-KCc-STn'`
- `M0_preprocess.py` with `study = 'MB-KCc-KCn'`
- `M1_generate_template.py`
- `M2_generate_masks.py`
- `M3_map_subtypes.py`

FLIM analysis scripts should be executed in this order:
- to be completed
