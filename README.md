# AI EU Act Assessment

This project is designed to classify AI systems according to the EU AI Act using a LLaMA 2 (7B) model. The AI system is evaluated through several stages, including **Initial Risk Assessment**, **Level-Based Risk Assessment**, and **Risk Categorization**. The purpose is to ensure that AI systems comply with regulations and are categorized according to their risk levels.

## Table of Contents
- [Project Description](#project-description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [AI Risk Assessment Workflow](#ai-risk-assessment-workflow)

## Project Description

This project evaluates AI systems to assess their compliance with the **EU AI Act**. The system follows a tree-based workflow, beginning with an **Initial Risk Assessment**. Depending on the results, it moves to **Level-Based Risk Assessment** or determines whether the AI system poses an **Unacceptable Risk**. The core components of the system include prompts designed for different phases of risk assessment and a fine-tuned LLaMA 2 model for generating responses.

## Features
- **AI Risk Categorization** based on EU AI Act principles.
- **LLaMA 2 (7B)** model integration for assessing and classifying risks.
- **Automated Workflow** for multi-step evaluation: Unacceptable risk detection, high-risk systems, and low-risk systems.
- **CSV-driven prompt system** for systematic AI assessment.

## Installation

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- Conda environment (optional but recommended)
- Hugging Face Token (required to access the LLaMA 2 model)

### Clone the Repository
```bash
git clone https://github.com/nglinhbao/AI-EU-Act-Assessment.git
cd AI-EU-Act-Assessment
```

### Set Up Virtual Environment (Optional)
```bash
conda create --name AI-Act python=3.10
conda activate AI-Act
```

### Install Required Packages
```bash
pip install -r requirements.txt
```

### Authenticate Hugging Face
Ensure you have a Hugging Face token to access the LLaMA 2 model.

```bash
huggingface-cli login
```

Or set the token directly:
```bash
export HUGGINGFACEHUB_API_TOKEN="your_token_here"
```

## Usage

![Workflow](./workflow.png)

1. **Run the Assessment Script:**

   To start the AI system classification process:
   ```bash
   python3 main.py
   ```

2. **Prompt System:**
   The system reads prompts from a CSV file (`ai_system_risk_prompts.csv`) and evaluates the AI system through different stages. If an AI system is deemed to pose an unacceptable risk, the process halts and returns the result. Otherwise, it proceeds through all stages to classify the system as high, limited, or minimal risk.

3. **Input and Output:**
   - The system evaluates inputs based on predefined prompts in the CSV file and generates a response for each stage using the LLaMA 2 model.
   - The output is the risk categorization of the AI system (e.g., **Unacceptable Risk**, **High Risk**, **Limited Risk**, or **Minimal Risk**).

## AI Risk Assessment Workflow

The assessment follows these key stages:

1. **Initial Risk Assessment**: 
   - The system checks for unacceptable risks like violations of human rights or major impacts on society.
   - If any prompt returns "Yes," the system categorizes the AI as **Unacceptable Risk** and halts further assessment.

2. **Level-Based Risk Assessment**: 
   - If the system passes the initial assessment, it moves to a more detailed evaluation of the AI's impact on health, safety, and data privacy.
   
3. **Risk Categorization**: 
   - The final stage assigns a risk level based on previous assessments, categorizing the AI system as **High Risk**, **Limited Risk**, or **Minimal Risk**.

4. **Ongoing Monitoring**: 
   - Additional prompts for continuous risk monitoring and ethical compliance
