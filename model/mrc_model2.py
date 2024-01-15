import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
#from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel

from transformers.models.roberta.modeling_roberta import *
_CHECKPOINT_FOR_DOC = "roberta-base"
_CONFIG_FOR_DOC = "RobertaConfig"
_TOKENIZER_FOR_DOC = "RobertaTokenizer"


class MRCQuestionAnswering(RobertaPreTrainedModel):
    config_class = RobertaConfig

    def _reorder_cache(self, past, beam_idx):
        pass

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # Use RobertaModel without pooling layer
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        words_lengths=None,
        start_idx=None,
        end_idx=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        span_answer_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward pass through RoBERTa
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        context_embedding = sequence_output

        # Compute align word sub_word matrix using a more concise approach
        align_matrix = self.compute_align_matrix(words_lengths, context_embedding)

        # Combine sub_word features to make word feature
        context_embedding_align = torch.bmm(align_matrix, context_embedding)

        logits = self.qa_outputs(context_embedding_align)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = self.compute_loss(
            start_logits, end_logits, start_positions, end_positions
        )

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def compute_align_matrix(self, words_lengths, context_embedding):
        # Compute align word sub_word matrix using a more concise approach
        batch_size = context_embedding.size(0)
        max_sub_word = context_embedding.size(1)
        max_word = max([len(sample_length) for sample_length in words_lengths])
        
        align_matrix = torch.zeros((batch_size, max_word, max_sub_word))

        for i, sample_length in enumerate(words_lengths):
            for j, length in enumerate(sample_length):
                start_idx = torch.sum(sample_length[:j])
                align_matrix[i, j, start_idx: start_idx + length] = 1 if length > 0 else 0

        align_matrix = align_matrix.to(context_embedding.device)
        return align_matrix

    def compute_loss(self, start_logits, end_logits, start_positions, end_positions):
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return total_loss
