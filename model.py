import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import paddlenlp as ppnlp


class QuestionMatching(nn.Layer):
    def __init__(self, pretrained_model, dropout=None, rdrop_coef=0.0):
        super().__init__()
        self.encoder = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
       
        # num_labels = 2 (similar or dissimilar)
        self.classifier = nn.Linear(self.encoder.config["hidden_size"], 2)
        self.rdrop_coef = rdrop_coef
        self.rdrop_loss = ppnlp.losses.RDropLoss()

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                ):

        _, cls_state = self.encoder(input_ids, token_type_ids, position_ids,
                                     attention_mask)
        cls_drop = self.dropout(cls_state)
        logits = self.classifier(cls_drop)
        outputs = (logits, )

        # For more information about R-drop please refer to this paper: https://arxiv.org/abs/2106.14448
        # Original implementation please refer to this code: https://github.com/dropreg/R-Drop
        if self.rdrop_coef > 0:
            _, cls_state2 = self.encoder(input_ids, token_type_ids,
                                         position_ids, attention_mask)
            cls_drop2 = self.dropout(cls_state2)
            rdrop_logits = self.classifier(cls_drop2)
            kl_loss = self.rdrop_loss(logits, rdrop_logits)
            co_logits = logits + rdrop_logits
            outputs += (kl_loss, co_logits)
        else:
            kl_loss = 0.0
            outputs += (kl_loss, )
        
        return outputs

class QuestionMatching_2stage(nn.Layer):
    def __init__(self, pretrained_model, dropout=None, rdrop_coef=0.0):
        super().__init__()
        self.encoder = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
       
        # num_labels = 2 (similar or dissimilar)
        self.classifier = nn.Linear(self.encoder.config["hidden_size"], 2)
        self.rdrop_coef = rdrop_coef
        self.rdrop_loss = ppnlp.losses.RDropLoss()

    def forward(self,
                input_ids,
                token_type_ids=None,
                input_ids2=None,
                token_type_ids2=None,
                position_ids=None,
                attention_mask=None,
                ):

        _, cls_state = self.encoder(input_ids, token_type_ids, position_ids,
                                     attention_mask)
        cls_drop = self.dropout(cls_state)
        logits = self.classifier(cls_drop)
        outputs = (logits, )

        # For more information about R-drop please refer to this paper: https://arxiv.org/abs/2106.14448
        # Original implementation please refer to this code: https://github.com/dropreg/R-Drop
        if self.rdrop_coef > 0:
            _, cls_state2 = self.encoder(input_ids2, token_type_ids2,
                                         position_ids, attention_mask)
            cls_drop2 = self.dropout(cls_state2)
            rdrop_logits = self.classifier(cls_drop2)
            kl_loss = self.rdrop_loss(logits, rdrop_logits)
            co_logits = logits + rdrop_logits
            outputs += (kl_loss, co_logits)
        else:
            kl_loss = 0.0
            outputs += (kl_loss, )
        
        return outputs

class MeasureLearing(nn.Layer):
    def __init__(self, pretrained_model, dropout=None, margin=1):
        super().__init__()
        self.encoder = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.margin = margin
        # num_labels = 2 (similar or dissimilar)
       
        

    def forward(self,
                input_ids1,
                input_ids2,
                label=None
                ):

        sequence_outputs1, cls_state1 = self.encoder(input_ids1)
        sequence_outputs2, cls_state2 = self.encoder(input_ids2)
        query1_state = sequence_outputs1.mean()
        query2_state = sequence_outputs2.mean()
        
        return query1_state, query2_state