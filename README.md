# Differentiating Motor and Cognitive Deficits in Neurological Disorders via a fixed-point iteration method

## Paper Summary

This repository contains the programming scripts, data, and analysis pipelines associated with the paper **"Mitigating the impact of motor impairment on self-administered digital tests in patients with neurological disorders."** In this study, the authors applied a computational framework to disentangle cognitive deficits from confounding motor impairments in self-administered digital tasks.

## Statistical techniques applied in the paper

Mathematical modelling, Mixed effects regressions modelling, Bayesian PCA analysis, Hypothesis Testing and Non-Parametric Methods

## Research in context

#### **Evidence before this study**

A PubMed search for 'neurological impairment', 'cognition', 'computerised testing', and 'remote testing' conducted in March 2025 highlighted growing interest in remote and computerised assessments over the past decade. The heterogeneity of cognitive deficits in neurological conditions and the debilitating effect these have on functional recovery highlights the benefit that computerised cognitive testing would have in terms of supporting more detailed assessments, early detection and longitudinal tracking of the impairments. However, these assessments typically exclude patients with co-occurring motor impairment, who would benefit the most from the remote nature of the assessments and from monitoring of symptoms. There is a need for inclusive methods that allow this large patient group to have their cognition assessed and monitored in a manner that is not confounded by motor impairment.

#### **Added value of this study**

Here we present a computational method that can reliably isolate cognitive ability from motor impairment in neurological patients. To test it, we chose stroke as a representative neurological disorder, as patients frequently experience both motor and cognitive impairments. We validated the framework across a very broad spectrum of cognitive domains, and across time. We found that it removes the confounding effect of motor impairment across all tasks, and that the resulting cognitive outcomes have stronger relationships with 1) established clinical assessments, 2) functional outcomes post-stroke and 3) MRI metrics of brain health.

#### **Implications of all the available evidence**

The current work shows that neurological patients with motor impairments no longer need to be excluded from remote clinical testing. Moreover, the current methodology can be applied to virtually any cognitive or digital test that records trial-level information, benefitting a wide range of neurological conditions. The current framework has the potential to reduce diagnostic bottlenecks, improve accessibility, and support early detection and intervention for a broad spectrum of neurological disorders.

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
