from model.mrc_model import MRCQuestionAnswering
from transformers import AutoTokenizer, pipeline
import torch
from nltk import word_tokenize

if __name__ == "__main__":
    model_checkpoint = "Phanh2532/XLMFinetuneQLora"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = MRCQuestionAnswering.from_pretrained(model_checkpoint)

    QA_input = {
        'question': "Thủ đô Việt Nam là ở đâu?",
        'context': "Thủ đô Việt Nam là Hà Nội."
    }

    while True:
        if len(QA_input['question'].strip()) > 0:
            inputs = [tokenize_function(QA_input, tokenizer)]
            inputs_ids = data_collator(inputs, tokenizer)
            outputs = model(**inputs_ids)
            answer = extract_answer(inputs, outputs, tokenizer)[0]
            print("answer: {}. Score start: {}, Score end: {}".format(answer['answer'],
                                                                      answer['score_start'],
                                                                      answer['score_end']))
        else:
            QA_input['context'] = input('Context: ')
        QA_input['question'] = input('Question: ')
