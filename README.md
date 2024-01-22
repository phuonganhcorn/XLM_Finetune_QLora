# XLM-Finetune-QLoRA for Vietnamese Language
#### Author: Ngo Phuong Anh


> [!NOTE]
> - This model is an optimized version of **XLMRoBERTa model** for finetuning it for QA task by **applied LoRA/QLoRA** into training process.
> - This project is inspired by a state-of-art **Mistral7B Large Language model** and a Finetuned XLMRoBERTa Version for Vietnamese Question Answering made by Nguyen Vu Le Binh.


### Overview
- **XLM-RoBERTa** is a pre-trained cross-lingual language model developed by Facebook AI. It combines elements from two popular models, XLM (Cross-Lingual Language Model) and RoBERTa (Robustly optimized BERT approach), to create a model that is capable of understanding and generating text in multiple languages. 
- The core architecture of XLM-RoBERTa is built based on Transformer architecture with Attention Mechanism. Made it better than other RNN, LTSM model for NLP tasks.
- **LoRA (Low Resource Adaptation) - QLoRA** encompasses two techniques designed to expedite the model fine-tuning process, facilitating users to train models efficiently on smaller GPU hardware.
- Presently, there exist several versions of Vietnamese QA models. The objective of this project is to create an optimized model by using XLM and QLoRA. This new model aims to enable users to efficiently fine-tune it for their specific tasks on smaller GPU hardware if desired.

### Requirements

Users can refer to the requirements.txt file for detailed information about the libraries required for this project or run this bash code below to install all needed libraries.
```bash
pip install -r requirements.txt
```
> [!TIP]
> 1. This project leverages the **Weights and Biases library** to automatically visualize training loss, evaluation loss, and store runtime information, streamlining the tracking process. *Users who only want to try interface of pre-trained model or don't need this can remove ```wandb``` from requirements.txt.*
> - Alternatively, I strongly recommend users who wish to fine-tune this model for their specific tasks to create a new ```wandb``` (Weights and Biases) account for saving their models.
> 2. Same with ```huggingface_hub``` library

### Dataset
#### 1. Information
In this project, the Dataset is combination of four large Vietnamese dataset which are
- Translated version of the Stanford Question Answering Dataset (SQuAD)
- UIT-ViQuAD (developed by UIT)
- MultiLingual Question Answering
- mailong25

#### 2. Handle Dataset
**_2.1_** By executing the provided scripts, users can process datasets and configure them into the required JSON format before feeding them into our model.
```bash
python utils/squad_to_mrc.py
```
**_2.2_** After process dataset, run this bash script below to split our dataset into train set and valid set
```bash
python utils/train_valid_split.py
```

The project focused on building a QA model for Vietnamese task. So the valid set is Vietnamese only 
In **total we have 15466 samples** and in my case, I split it with 10% (this is optional, users can change this percentage to their preference) into 2 train and valid set. This will make
**- Train set: 13919 samples**
**- Valid: 1547 samples**

