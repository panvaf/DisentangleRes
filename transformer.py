"""
Transformer classes.
"""

# Import box
import torch
from torch import nn
from transformers import GPT2Config, GPT2PreTrainedModel, GPT2Model
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from typing import Optional, Tuple, Union
    

class GPT2ContinuousInputs(GPT2PreTrainedModel):
    """
    GPT2 model with additional linear input layer and without output layer.
    
    The model skips the embedding/tokenization steps and projects from
    directly from inputs to hidden state.
    
    The positional embeddings are omitted, since the inputs are assumed iid.

    Args:
        config (`GPT2Config`): Model configuration class with all the parameters of the model.
    """
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        # Define the input projection layer
        self.input_projection = nn.Linear(config.n_in, config.n_embd, bias=True)
        # Initialize weights using the model's method
        self._init_weights(self.input_projection)
        # Apply final processing
        self.post_init()
        

        # Initialize positional embeddings to zero and make them non-differentiable
        with torch.no_grad():
            self.transformer.wpe.weight.zero_()
        self.transformer.wpe.weight.requires_grad = False
        

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor, ...], ...]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,  # Used for continuous inputs
        labels: Optional[torch.FloatTensor] = None,  # Vector labels per token
        label_mask: Optional[torch.BoolTensor] = None,  # Boolean mask indicating which labels to consider
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        """"
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, embedding_dim)`, *optional*):
                Optionally, instead of passing `input_ids`, you can choose to directly pass an embedded representation.
                If the embeddings are of size 40 (your continuous inputs), they will be projected to the model's hidden size.
            labels (`torch.FloatTensor` of shape `(batch_size, sequence_length, n_task)`, *optional*):
                Vector labels for computing the regression loss.
            label_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Boolean mask indicating which labels to consider in the loss computation.
                `True` for tokens to include in loss, `False` to ignore.
            All other arguments are the same as in `GPT2Model`.

        Returns:
            `VectorOutput` or a tuple, depending on `return_dict`.
        """
        # Set default value for return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Handle inputs_embeds
        if inputs_embeds is not None:
            if inputs_embeds.size(-1) == self.config.n_in:
                # If inputs_embeds are continuous inputs of size 40, project them
                inputs_embeds = self.input_projection(inputs_embeds)
            else:
                raise ValueError(
                    f"inputs_embeds last dimension ({inputs_embeds.size(-1)}) "
                    f"does not match input dimension ({self.config.n_in})"
                )
            input_ids = None  # Ensure input_ids is None when using inputs_embeds
        else:
            if input_ids is None:
                raise ValueError("You must provide either input_ids or inputs_embeds")


        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,  # Force return_dict to True
        )

        return transformer_outputs