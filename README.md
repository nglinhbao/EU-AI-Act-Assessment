# EU AI Act Assessment

This project is designed to classify AI systems according to the **EU AI Act** using the **LLaMA 2 (7B)** model. The AI system is evaluated through several stages, including **Initial Risk Assessment**, **Level-Based Risk Assessment**, and **Risk Categorization**. The purpose is to ensure that AI systems comply with regulations and are categorized according to their risk levels.

## Table of Contents
- [Project Description](#project-description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [AI Risk Assessment Workflow](#ai-risk-assessment-workflow)

## Project Description

This project evaluates AI systems to assess their compliance with the **EU AI Act**. The system follows a tree-based workflow, beginning with an **Initial Risk Assessment**. Based on the outcomes, it moves to **Level-Based Risk Assessment** or determines whether the AI system poses an **Unacceptable Risk**. The core components of the system include structured prompts designed for different phases of risk assessment and a fine-tuned **LLaMA 2** model to generate responses.

## Features
- **AI Risk Categorization** based on **EU AI Act** principles.
- **LLaMA 2 (7B)** model integration for automated risk assessments.
- **CSV-driven prompt system** for systematic AI evaluation.
- **Multi-step automated workflow**: Initial risk, level-based assessment, and ongoing monitoring.

## Installation

### Prerequisites

Ensure you have the following installed:
- Python 3.8+.
- Conda environment (optional but recommended).
- Hugging Face API Token (required to access the LLaMA 2 model).

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

Alternatively, you can export the token directly in your environment:
```bash
export HUGGINGFACEHUB_API_TOKEN="your_token_here"
```

## Usage

### Steps to Run the Assessment:

1. **Prepare the AI System Description**:
   - Create a `.txt` file containing the description of the AI system that you wish to assess.
   - Example:
     ```txt
     This AI system assists in medical diagnostics by analyzing patient health data. It processes sensitive health records and suggests possible conditions based on the data.
     ```

2. **Run the Assessment Script**:

   To start the AI system classification process, make sure you have the system description `.txt` file ready, and execute the following command:

   ```bash
   python3 main.py
   ```

3. **Prompts and AI System Evaluation**:
   - The system reads prompts from the `CSV` file (`ai_system_risk_prompts.csv`) and evaluates the AI system based on its description. 
   - If the AI system poses an **Unacceptable Risk**, the process halts and returns the result. Otherwise, it proceeds through all stages to classify the system as **High Risk**, **Limited Risk**, or **Minimal Risk**.

4. **Output**:
   - The system provides a risk categorization based on the predefined prompts in the CSV file.
   - Example outcomes include: **Unacceptable Risk**, **High Risk**, **Limited Risk**, or **Minimal Risk**.

### Example Output:

Here is an example of how the system would respond to various prompts:

```plaintext
Prompt: Does the AI system violate fundamental human rights or Union values?
Response: System Description: This AI system is designed to assist in medical diagnostics by analyzing patient health data and suggesting possible conditions. It uses a combination of supervised and unsupervised learning models. Sensitive data such as patient health records and biometric data are processed.

Answer: The AI system does not violate fundamental human rights or Union values.

### 1. Does the AI system violate the human right to privacy?
The AI system does not violate the human right to privacy.

### 2. Does the AI system violate the human right to non-discrimination?
The AI system does not violate the human right to non-discrimination.

### 3. Does the AI system violate the human right to equal treatment?
The AI system does not violate the human right to equal treatment.

### 4. Does the AI system violate the human right to data protection?
The AI system does not violate the human right to data protection.

### 5. Does the AI system violate the right to effective remedy?

Prompt: Does the AI system impact human life (e.g., health, safety, security)? Explain how the system mitigates risks in these areas.
Response: System Description: This AI system is designed to assist in medical diagnostics by analyzing patient health data and suggesting possible conditions. It uses a combination of supervised and unsupervised learning models. Sensitive data such as patient health records and biometric data are processed.

### 1. System Description
#### 1.1. System Goals
The AI system's primary goal is to provide accurate and timely diagnoses for patients. It achieves this by analyzing patient health data and suggesting possible conditions. The system also provides information about the diagnosis and treatment options to the patients.

#### 1.2. System Architecture
The AI system is designed to be scalable and modular. It consists of a centralized server that receives patient health data and sends it to a cloud-based analytics engine. The analytics engine processes the data and provides the diagnosis and treatment options to the patients.

#### 1.3. System Components
The AI system consists of the following components:
- Centralized server: This component receives patient health data and sends it to the cloud-based analytics engine.
- Analytics engine: This component processes the data and generates the diagnosis.

Prompt: Does the AI system handle sensitive personal data such as biometrics, health information, or financial records? If yes, how is data security ensured?
Response: System Description: This AI system is designed to assist in medical diagnostics by analyzing patient health data and suggesting possible conditions. It uses a combination of supervised and unsupervised learning models. Sensitive data such as patient health records and biometric data are processed.

Answer: The AI system does handle sensitive personal data such as biometrics, health information, or financial records. Data security is ensured through the use of encryption and secure storage. The system also has access control measures in place to prevent unauthorized access to sensitive data.

Classified as: High Risk
The AI system is classified as: High Risk.
```

## AI Risk Assessment Workflow

![Workflow](./workflow.png)

The assessment follows these key stages, evaluated in sequence from top to bottom:

1. **Unacceptable Risk**:
   - The system is first checked against prompts to identify if it violates fundamental human rights, Union values, or carries risks that are deemed unacceptable.
   - If any prompt receives a "Yes" answer, the system is classified as **Unacceptable Risk**, and the assessment halts. No further evaluation is carried out.

2. **High Risk**:
   - If the system passes the Unacceptable Risk stage, it is checked for **High Risk** factors. This includes assessing whether the system impacts human life, handles sensitive personal data, or makes autonomous decisions without human intervention.
   - If any prompt in this stage receives a "Yes" answer, the system is classified as **High Risk**.

3. **Limited Risk**:
   - If the system does not qualify for **High Risk**, it is evaluated for **Limited Risk**. Prompts in this stage assess bias detection, transparency, and fairness in the system.
   - If any prompt receives a "Yes" answer, the system is classified as **Limited Risk**.

4. **Minimal Risk**:
   - If none of the previous stages result in a classification, the system is classified as **Minimal Risk**. This means the system poses minimal risks in terms of data handling, bias, and overall impact on users and society.

5. **Ongoing Monitoring**:
   - Regardless of the initial classification, ongoing monitoring prompts help ensure the system remains compliant with performance, ethical, and fairness standards over time. This stage includes provisions for continued oversight and assessment to catch potential issues post-deployment.
