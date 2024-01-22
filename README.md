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
> [!CAUTION]

