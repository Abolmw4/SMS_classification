import torch.nn as nn
import torch


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bertmodel = torch.hub.load('huggingface/pytorch-transformers', 'modelForSequenceClassification',
                                        'bert-base-cased-finetuned-mrpc')
        self.bertmodel.bert.embeddings.word_embeddings.weight.requires_grad = False
        self.bertmodel.bert.embeddings.position_embeddings.weight.requires_grad = False
        self.bertmodel.bert.embeddings.token_type_embeddings.weight.requires_grad = False
        self.bertmodel.bert.embeddings.LayerNorm.weight.requires_grad = False
        self.bertmodel.bert.embeddings.LayerNorm.bias.requires_grad = False
        self.act = nn.Softmax(dim=1)
        i = 0
        for name, param in self.bertmodel.bert.encoder.layer.named_parameters():
            if name == '10.attention.self.query.weight':
                break
            else:
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        x = self.bertmodel(input_ids, attention_mask)
        output = self.act(x['logits'])
        return output
