# **Using Semantic Entropy Probes to Efficiently Steer Away from LLM Hallucinations**  
**CS230 Course Project by Walid Rahman, Alex Popa, and Dilan Nana**  

---

## **Overview**  
This project explores semantic entropy probes (SEPs) applied to the Llama2-7b language model. It includes:  
- Integration with the **Llama2-7b model**.  
- Use of **Semantic Entropy Probes (SEPs)**, precomputed for specific token positions.  
- Dataset prompts and evaluations using **SQuAD v2**.  
- Comprehensive analysis using the **SimpleQA Evaluator** to study model behavior.  

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
   - **OpenAI API Key**: Set your OpenAI API key in `run_probes.py`.  

---

## **Main Notebook**  
The primary entry point for this project is the `run_probes.ipynb` notebook. Follow these steps to run the project:  

### **1. Loading Precomputed Probes**  
Precomputed SEPs are provided for **SLT** and **TBG** token positions:  
- `model_dict_slt_ent.pkl`  
- `model_dict_tbh_ent.pkl`  

These files, sourced from [Kossen et al.](https://github.com/OATML/semantic-entropy-probes), have been reformatted for ease of use. Each file contains probes for all layers of the **Llama2-7b model**.

### **2. Data Preparation**  
- Load the **SQuAD v2 dataset** and generate prompts following the styles used in the SEP construction.  

### **3. Model Setup**  
- Load the **Llama2-7b model** and initialize the **SimpleQA Evaluator**.  
- Load the **Semantic Entropy Probes (SEPs)**.  

### **4. End-to-End Flow**  
- Generate answers for 1,000 samples from the dataset.  
- Capture hidden states at **TBG** and **SLT** positions.  
- Store SEP outputs and generated answers in a **Pandas DataFrame**.  

**Note:** Results are stored every 20 rows to prevent data loss and ensure smooth operation.

---

## **References**  
This project leverages the work of Kossen et al. on **Semantic Entropy Probes**:  
- Repository: [OATML Semantic Entropy Probes](https://github.com/OATML/semantic-entropy-probes).  

For questions or contributions, please reach out to the project team:  
- Walid Rahman  
- Alex Popa  
- Dilan Nana  

--- 

## **License**  
This project is for educational purposes as part of **CS230 coursework**. Please respect the terms of usage for any external tools and datasets.

---

