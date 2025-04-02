# Differentiating Motor and Cognitive Deficits in Neurological Disorders via a fixed-point iteration method

## Executive Summary

This repository contains the data cleaning and analysis pipelines associated with the paper **"Mitigating the impact of motor impairment on self-administered digital tests in patients with neurological disorders."** In this project, I developed a computational framework to improve the reliability and clinical utility of remote digital health assessments in patients struggling with physical/motor impairments. The results suggest that the negative impact of physical impairment on performance has been fully mitigated and that the framework yields highly reliable and clinically valid outcomes.

## Statistical techniques applied in the paper

Mathematical modelling, Mixed effects regressions modelling, Bayesian PCA analysis, Hypothesis Testing and Non-Parametric Methods

## Research in context

#### **Evidence before this study**

There is growing interest in developing remote, computerised cognitive assessments for both clinical and research use. The heterogeneity of cognitive deficits in neurological conditions and their detrimental impact on functional outcomes, underscore the value of computerised cognitive testing in facilitating early detection and longitudinal tracking of the impairments. However, these self-administered assessments are confounded by commonly occurring motor deficits. Consequently, these vulnerable patients, who may stand to benefit the most from the remote nature of the assessments, are often excluded from participating in them. A PubMed search for ‘neurological impairment’, ‘cognition’, ‘remote testing’, and ‘computerised testing’ on 10 March 2025, revealed that this was not addressed in the literature to date. There is a need for developing inclusive methods that enable accurate cognitive assessment and monitoring for this large patient group, free from the confounding effects of motor impairment.

#### **Added value of this study**

Here, I apply a computational method that can reliably isolate true cognitive ability from hand-motor impairment in neurological patients. Using patients with stroke as a representative neurological disorder affecting both motor and cognitive impairments, I validated the framework across a broad spectrum of cognitive domains and recovery stages. I show that the model removes the confounding effect of hand motor impairment across all digital tasks, and that the resulting cognitive outcomes have stronger relationships with 1) established clinical assessments, 2) quality of life and independence post-stroke and 3) MRI measures of brain injury.

#### **Implications of all the available evidence**

Neurological patients experiencing motor impairments can now benefit from self-administered remote cognitive assessments, which, when enhanced by our computational framework, yield reliable and clinically valid measurements. Furthermore, this approach has a wide applicability to any cognitive task or digital test that records trial-by-trial performance information, potentially benefitting a wide range of digital health initiatives. By reducing diagnostic bottlenecks, enhancing accessibility to vulnerable patient groups, and enabling early detection and monitoring of deficits, the current framework holds significant potential for improving care across a broad spectrum of neurological conditions.

<p align="center">

<img src="Figures/key tasks graph.png"/>

**Figure 1** - Four representative digital tasks discussed in the paper, out of 18, with opposing levels of motor and cognitive load. Each task is \~3 minutes in duration. Highlighted text refers to the primary cognitive domain being assessed.

</p>

## Repository Structure

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

Dependencies for the Python and R scripts can be found and installed from the `Docs/` Directory

3.  **Data Download**

The data can be made available on reasonable request and upon institutional regulatory approval. For more information email [dragos-cristian.gruia19\@imperial.ac.uk](mailto:dragos-cristian.gruia19@imperial.ac.uk){.email}

## Running the Analyses

1.  **Data Preparation:** Ensure that all required datasets are placed in a directory called data/

2.  **Execute Analysis Scripts** Run the main analysis script to reproduce the results:

    For the Python scripts use `Python_scripts/main.py` to run the entire analysis pipeline. The only information required to run the function are the path to the data and the desired output directory. See examples and further details inside the function.

    For the R scripts use `R_scripts/run_all.R` to run all scripts and save the outputs.

3.  **Output** Generated figures are saved in Figures/directory

## Citation

If you use this repository in your work, please cite the paper as follows:

`Gruia, D.-C., Giunchiglia, V., Braban, A., Parkinson, N., Banerjee, S., Kwan, J., Hellyer, P. J., Hampshire, A., & Geranmayeh, F. (2025). Mitigating the impact of motor impairment on self-administered digital tests in patients with neurological disorders. MedRxiv. https://doi.org/10.1101/2025.03.14.25323903`

## License

This project is licensed under the MIT License. See the LICENSE file for details.
