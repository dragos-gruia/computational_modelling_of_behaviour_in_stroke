# Differentiating Motor and Cognitive Deficits in Neurological Disorders via a fixed-point iteration method

## Paper Summary

This repository contains the programming scripts, data, and analysis pipelines associated with the paper **"Differentiating motor and cognitive deficits in patients with neurological impairment during self-administered digital tasks."** In this study, the authors applied a computational framework to disentangle cognitive deficits from confounding motor impairments in self-administered digital tasks. Key highlights include:

-   **Objective:**\
    To isolate cognitive performance from motor impairments (such as those caused by an impaired hand or device variability) in neurological patients, via an iterative modelling framework (IDoCT).

-   **Methods:**\
    The study analyzed data from 171 stroke survivors (assessed longitudinally) alongside a normative cohort of 6000+ healthy older adults. The framework computes a modelled *Cognitive Index* by triangulating trial accuracy, reaction time, and trial difficulty, while removing motor-related confounds.

-   **Results:**\
    Standard cognitive performance metrics were significantly affected by motor impairments and by the device that patients used. In contrast, the modelled Cognitive Index showed no significant confounding effects and demonstrated stronger correlations with pen-and-paper clinical scales (e.g., MoCA), quality of life post-stroke (IADL), and neuroimaging metrics of cerebrovascular disease.

-   **Conclusion:**\
    The modelled Cognitive Index improves the clinical utility and validity of remote digital assessments in patients with co-occurring motor and cognitive deficits. This approach has the potential to reduce diagnostic delays and enhance early detection and intervention strategies across a range of neurological disorders.

## Repository Structure

├── Data_availability/ - Information on how to obtain the data used in this paper

├── Python_scripts/ - Analysis scripts in Python

├── R_scripts/ - Analysis scripts and Jupyter notebooks

├── Figures/ - Generated figures and visualizations

├── Docs/ - Additional information on which dependencies to install

├── README.md - This file

└── LICENSE - License information

## Requirements

-   **Programming Languages:** Python 3.10.5 and R 4.3.1

## Installation

1.  **Clone the Repository:**

`bash git clone <https://github.com/dragos-gruia/computational_modelling_of_behaviour_in_stroke>  cd repository-name`

2.  **Install Dependencies:**

`bash pip install -r docs/requirements.txt`

3.  **Data Download**

The data can be made available on reasonable request and upon institutional regulatory approval. For more information email [dragos-cristian.gruia19\@imperial.ac.uk](mailto:dragos-cristian.gruia19@imperial.ac.uk){.email}

## Running the Analyses

1.  **Data Preparation:** Ensure that all required datasets are placed in the data/ directory.

2.  **Execute Analysis Scripts** Run the main analysis script to reproduce the results:

    `bash python scripts/run_analysis.py`

3.  **Output** Generated figures are saved in figures/directory

## Citation

If you use this repository in your work, please cite the paper as follows:

`Gruia, D.-C., Giunchiglia, V., Braban, A., Parkinson, N., Banerjee, S., Kwan, J., Hellyer, P. J., Hampshire, A., & Geranmayeh, F. (2025). Differentiating motor and cognitive deficits in patients with neurological impairment during self-administered digital tasks. Pre-print.`

## License

This project is licensed under the MIT License. See the LICENSE file for details.
