# Finetune XLMRoBERTa with QLoRA for Vietnamese QA task
#### Author: Ngo Phuong Anh

> [!NOTE]
> - This model is an optimized version of **XLMRoBERTa model** for QA task by **applied LoRA/QLoRA** into training process.
> - This project is inspired by a state-of-art **Mistral7B Large Language Model** and a Finetuned XLMRoBERTa Version for Vietnamese Question Answering Model made by Nguyen Vu Le Binh.


### OVERVIEW
- **XLM-RoBERTa** is a pre-trained cross-lingual language model developed by Facebook AI. It combines elements from two popular models, XLM (Cross-Lingual Language Model) and RoBERTa (Robustly optimized BERT approach), to create a model that is capable of understanding and generating text in multiple languages. 
- The core architecture of XLM-RoBERTa is built based on Transformer architecture with Attention Mechanism. Made it better than other RNN, LTSM model for NLP tasks.
- **LoRA (Low Rank Adaptation) - QLoRA** encompasses two techniques designed to expedite the model fine-tuning process, facilitating users to train models efficiently on smaller GPU hardware.
- Presently, there exist several versions of Vietnamese QA models. The objective of this project is to create an optimized model by using XLM and QLoRA. This new model aims to enable users to efficiently fine-tune it for their specific tasks on smaller GPU hardware if desired.

### REQUIREMENTS

Users can refer to the requirements.txt file for detailed information about the libraries required for this project or run this bash code below to install all needed libraries.
```bash
pip install -r requirements.txt
```
> [!TIP]
> 1. This project leverages the **Weights and Biases library** to automatically visualize training loss, evaluation loss, and store runtime information, streamlining the tracking process. *Users who only want to try interface of pre-trained model or don't need this can remove ```wandb``` from requirements.txt.*
> - Alternatively, I strongly recommend users who wish to fine-tune this model for their specific tasks to create a new ```wandb``` (Weights and Biases) account for saving their models.
> 2. Same with ```huggingface_hub``` library

### DATASET
#### 1. Information
In this project, the Dataset is a combination of four large Vietnamese datasets which are
- Translated version of the Stanford Question Answering Dataset (SQuAD)
- UIT-ViQuAD (developed by UIT)
- MultiLingual Question Answering
- mailong25

For a more detailed view, users can refer to the contents in the ```data-bin``` folder.

#### 2. Handle Dataset
##### 2.1. Config data 
By executing the provided scripts, users can process datasets and configure them into the required JSON format before feeding them into our model.
```bash
python utils/squad_to_mrc.py
```

##### 2.2. Split data 
After process dataset, run this bash script below to split our dataset into train set and valid set
```bash
python utils/train_valid_split.py
```

The project focused on building a QA model for Vietnamese task. So _the valid set is Vietnamese only_ 

In **total we have 15466 samples** and in my case, I split it with 10% (this is optional, users can change this percentage to their preference) into 2 train and valid set. This will make

- **Train set: 13919 samples**
- **Valid: 1547 samples**

#### 3. FINETUNE MODEL
##### 3.1. Raw model with no QLoRA
```bash
python main.py
```

This bash script will train model withouth apply QLoRA on it. 

##### 3.2. Finetune model with QLoRA
```bash
python mainQLora.py
```
This bash script will train model with QLoRA on it.

> [!NOTE]
> By running ```main``` file, we run other file which are```utils/data_loader.py``` (For load data into base model) and ```model/mrc_model.py``` (Load base model).

#### 4. Inference
After training process, users can run
```bash
python infer.py
```
This will allow us to run model we just trained.

#### 5. PRE-TRAINED
For users who only want to run inference, can run this code
```py
# test model on hugging face

from transformers import pipeline
model_checkpoint = "Phanh2532/XLMQLoraCustom"
nlp = pipeline('question-answering', model=model_checkpoint,
                   tokenizer=model_checkpoint)
QA_input = {
  'question': "Một năm có bao nhiêu tháng có 31 ngày?",
  'context': "8 tháng"
}
res = nlp(QA_input)
print('pipeline: {}'.format(res))
```
This is the model I that already applied QLoRA and finetuned it with the dataset for Vietnamese QA task that we prepared before.

#### Training model on Google Colab
For users who want to finetune model on Google Colab, can use 2 files which are ```XLMQLoraMRC.ipynb``` and ```XLMFinetune_raw.ipynb```
> [!CAUTION]
> - With ```XLMFinetune_raw.ipynb```, because this is the version that I didn't apply QLoRA on. The maximum epochs that this model can run on free T4 15GB GPU on Google Colab is 4 epochs (recommend 3 for inference later).
> - ```XLMQLoraMRC.ipynb``` can run up to 7 epochs.
