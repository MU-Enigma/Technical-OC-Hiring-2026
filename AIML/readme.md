# AI/ML Head

## Task

Build a deep learning model for image upscaling (super-resolution) using only a folder of ordinary target-resolution images.

You do not have access to a premade low-res/high-res dataset, and you cannot download one. All you have is a folder of ordinary target-resolution images - you may use any source, between 300 and 1000 images. There is no X (input) without you creating it. Your task is to construct the training pairs yourself, justify how you did it, and use them to actually train a model.

---

## Submission Requirements

Your submission should include:

### 1. Data and Degradation Strategy

- Describe how you generate the low-res input from each high-res target
- Explain the chosen degradation strategy and why it is appropriate
- Discuss the scale factor and how it affects training and generalization
- Explain what happens if the degradation is too clean or too simple
- Describe how you split the data and how you avoided leakage

### 2. Model Design

- Explain your model architecture and justify the design choice
- Discuss the loss function you used and why it is suitable for this task
- Compare your chosen approach with at least one alternative you considered

### 3. Evaluation and Analysis

- Explain what metrics you use to judge upscaling quality
- Show how you evaluate the model beyond just loss values
- Include a comparison between a baseline model and your final model

---

## Key Deliverables

Your submission should include:

- Working code covering data setup, generation of train/val samples, the model, training loop, and evaluation
- Training progression images showing input, target, and model output at regular intervals
- A short write-up covering the points above
- A README and requirements file so the project can be run cleanly
- Model configuration files in a structured format such as JSON
- Separate scripts or entry points for each trained model configuration
- A short demo video explaining the implementation, reasoning, and results if possible
- A junior-level workshop-style presentation on the project
- A technical video walkthrough explaining your approach, implementation, and key design decisions
- A README.md technical document included in your GitHub submission

---

## Success Criteria

Evaluation will be based on:

- Technical understanding and reasoning behind your design choices
- Quality of research and justification for the chosen methods
- Soundness of the architecture and loss-function decisions
- Quality of evaluation and comparison between models
- Replicability of the code and results
- Clarity of documentation and presentation

---

## Presentation

Prepare a short presentation or walkthrough explaining your approach, your reasoning, your results, and how your model was trained and evaluated.
