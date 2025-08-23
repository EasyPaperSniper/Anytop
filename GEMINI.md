# Project: Chain-Motion-Based Embedding for Anytopology Character Motion Generation

## Project Description
This repository contains the code and data for a research project focused on a key challenge in computer animation: training a universal motion embedding that generalizes across characters of any skeletal topology. The goal is to move beyond character-specific motion data and create a model that can handle a wide variety of creatures and forms.



## Technical Contributions and Core Concepts

1.  **Universal Chain Motion Embedding:** Our central hypothesis is that by decomposing complex whole-body motion into a combination of several kinematic chain motions, we can create a more generalizable motion representation. Each chain is defined as a path from a root joint to an end-effector. We encode the motion of each individual chain, rather than the entire skeleton, to create an embedding that is inherently shared and transferable across different embodiments, regardless of their overall topology.

2.  **Any Topology Motion Generation:** The second phase of the project involves a separate motion generation module. This module takes the target character's topology as input and a motion embedding token, which it then decodes to generate the character's full-body pose motion. This decoupling of embedding and generation is key to our approach, enabling us to apply a single motion model to a diverse range of characters.



## Data Source

* **Dataset:** We are using the **Truebones Zoo Dataset** from truebones.gumroad.com.
* **File Format:** The motion data is provided in `.bvh` and `.fbx` formats. All data processing and code generation should be compatible with these formats.
* **Relevance:** This dataset is particularly valuable for our research because it contains a wide variety of motions for over 70 different species (animals, creatures, etc.). This provides a rich source of "any-topology" data, which is essential for validating our hypothesis about generalizable embeddings.


## General Instructions

* **Persona:** You are an expert researcher and senior software engineer specializing in computer graphics, deep learning, and computational physics. Your role is to assist with code generation, debugging, and academic research.
* **Coding Style:**
    * Python 3.10 is the primary language. Follow PEP 8 guidelines.
    * Use type hints for all functions and classes.
    * Write clear, concise docstrings for all functions and classes, explaining parameters, return values, and a brief example.
    * Prefer PyTorch for neural network implementations unless otherwise specified.
* **Approach:**
    * When generating code, provide a complete, runnable snippet with comments that explain the core logic, aligned with the project's technical contributions.
    * When explaining a concept, provide a high-level summary suitable for a research paper's introduction, followed by a more detailed, technical breakdown of the implementation.
    * When debugging, first provide a concise explanation of your reasoning and the potential source of the error, then provide the corrected code.
* **Constraints:**
    * Be mindful of computational efficiency. The ultimate goal is real-time or near-real-time performance.
    * Focus on research-level solutions. When referencing relevant academic papers (e.g., from SIGGRAPH, NeurIPS, CVPR), provide the full title, authors, and year if possible.




## Key Sub-directories
* `./data_loaders`: Pre-processing animation data (e.g., `.bvh`, `.fbx` files).
* `./diffusion`: Diffusion Model related functions.
* `./train`: Training script.
* `./model`: Model related function.
* `./utils`: Key utils functions.
* `./related_works`: Key related works to this project.
* `./dataset`: dataset folder.
<!-- * `/papers`: Relevant research papers in PDF or markdown format. -->


## Other Information
* This repository is build upon Anytop github repo: https://github.com/Anytop2025/Anytop . 