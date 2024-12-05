# **Using Semantic Entropy Probes to Efficiently Steer Away from LLM Hallucinations**  
**CS230 Course Project by Walid Rahman, Alex Popa, and Dilan Nana**  

---

## **Overview**  
This project explores semantic entropy probes (SEPs) applied to the Llama2-7b language model. It includes:  
- Integration with the **Llama2-7b model**.  
- Use of **Semantic Entropy Probes (SEPs)**, precomputed for specific token positions.  
- Dataset prompts and evaluations using **SQuAD v2**.
- Steering of models using steering vectors
- Comprehensive analysis using the **our custom scorerr** to study model behavior.
- **KL MODEL TO GUIDE MODEL STEERING BASED ON SEPs**

---

## **Setup Instructions**

1. **Install Dependencies**  
   Install required packages using the provided `requirements.txt` file:  
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

2. **Prerequisites**  
   - **Model Access**: Ensure you have access to the **Llama2-7b model**.  
   - **Hardware**: A **GPU** is required to run the model efficiently.  
   - **OpenAI API Key**: Please supply your own key.  

---

## **Main Notebook**  
First run the generate_answers_run_probes.ipynb. Then run KL_model.ipynb.

### **1. Download Precomputed Probes and Steering Vectors**  
Precomputed SEPs are provided for **SLT** and **TBG** token positions:  
- `model_dict_slt_ent.pkl`  
- `model_dict_tbh_ent.pkl`  

These files contains the probes trained via the notebook provided by  [Kossen et al.](https://github.com/OATML/semantic-entropy-probes). Each file contains probes for all layers of the **Llama2-7b model**.

To download the Steering vectors, please use the [CAA repository](https://github.com/nrimsky/CAA). 

### **2. Data Preparation**  
- Load the **SQuAD v2 dataset** and generate prompts following the styles used in the SEP construction. This is done in generate_answers_run_probes.ipynb.

### **3. Model Setup**  
- Load the **Llama2-7b model** 
- Load the **Semantic Entropy Probes (SEPs)**.  
- Load the steering vectors

### **4. End-to-End Flow**  
- Run generate_answers_run_probes.ipynb to generate answers from LLMs using steering vectors and also run the SEP probes.
- Run the KL_model.ipynb to train the KL model to guide steering


NOTE: the milestone folder contains code from the milestone

---

## **References**  
This project leverages the work of Kossen et al. on **Semantic Entropy Probes**:  
- SEP Repository: [OATML Semantic Entropy Probes](https://github.com/OATML/semantic-entropy-probes).  
- CAA Repository: (https://github.com/nrimsky/CAA)

For questions or contributions, please reach out to the project team:  
- Walid Rahman  
- Alex Popa  
- Dilan Nana  

--- 

## **License**  
This project is for educational purposes as part of **CS230 coursework**. Please respect the terms of usage for any external tools and datasets.

---
