# test model on huggin face

from transformers import pipeline
# model_checkpoint = "nguyenvulebinh/vi-mrc-large"
model_checkpoint = "Phanh2532/XLMQLoraCustom"
nlp = pipeline('question-answering', model=model_checkpoint,
                   tokenizer=model_checkpoint)
QA_input = {
  'question': "Một năm có bao nhiêu tháng có 31 ngày?",
  'context': "8 tháng"
}
res = nlp(QA_input)
print('pipeline: {}'.format(res))
