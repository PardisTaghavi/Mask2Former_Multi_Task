import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec




@BACKBONE_REGISTRY.register()
class MoCoV3Backbone(Backbone):
    def __init__(self, base_encoder, dim=256, K=65536, m=0.99):
        super().__init__()
        self.K = K  # Queue size
        self.m = m  # Momentum

        self.encoder_q = base_encoder  # Query encoder
        self.encoder_k = copy.deepcopy(base_encoder)  # Momentum encoder
        for param in self.encoder_k.parameters():
            param.requires_grad = False

        for param in self.encoder_k.parameters():
            param.requires_grad = False  # Stop gradient in momentum encoder

        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Projection Head (MLP with BatchNorm)
        self.projection_head_q = nn.Sequential(
            nn.Linear(768, 256, bias=False),  # Ensure this matches backbone output
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256, bias=False)
        )
        self.projection_head_k = copy.deepcopy(self.projection_head_q)
        for param in self.projection_head_k.parameters():
            param.requires_grad = False

        #assert param.requies grad = Trie for param in self.projection_head_q.parameters()
        for param in self.projection_head_q.parameters():
            assert param.requires_grad == True

        # Ensure output features are set correctly
        self._out_features = getattr(base_encoder, "_out_features", list(base_encoder.out_features))
        self._out_feature_channels = base_encoder._out_feature_channels
        self._out_feature_strides = base_encoder._out_feature_strides

    @property
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data.detach()
        for param_q, param_k in zip(self.projection_head_q.parameters(), self.projection_head_k.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data.detach()

   
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Handle cases where batch size is not a divisor of K
        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            num_wrap = (ptr + batch_size) % self.K
            self.queue[:, ptr:] = keys[:batch_size - num_wrap].T
            self.queue[:, :num_wrap] = keys[batch_size - num_wrap:].T

        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr
        # print("Queue ptr:", ptr)                # anumber between 0 and K=65536
        # print("Queue shape:", self.queue.shape) #  (256, 65536)




    def forward(self, x_q, x_k):
        self._momentum_update_key_encoder()

        # Extract features
        features_q = self.encoder_q(x_q)
        features_k = self.encoder_k(x_k)

        if "res5" not in features_q or "res5" not in features_k:
            raise KeyError("Expected 'res5' key in backbone features but got:", features_q.keys())

        q_moco = features_q["res5"] #4, 768, 19, 37
        k_moco = features_k["res5"] #4, 768, 19, 37

        # Flatten and pool
        q_moco = q_moco.flatten(2).mean(dim=-1) #4, 768
        k_moco = k_moco.flatten(2).mean(dim=-1) #4, 768


        # Projection head
        q_moco = self.projection_head_q(q_moco)
        k_moco = self.projection_head_k(k_moco)

        # Normalize
        q_moco = F.normalize(q_moco, dim=1)
        k_moco = F.normalize(k_moco, dim=1)

        # Enqueue normalized k_moco
        self._dequeue_and_enqueue(k_moco.detach())

        return features_q, q_moco, k_moco
