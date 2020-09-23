# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


class Sim2Sem(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, conv_encoder, fea_dim, num_clusters, K=65536, m=0.999, T=0.07, sr=0.8, in_channels=3, freeze_encoder_s=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: sim2sem momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(Sim2Sem, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        self.sr = sr
        self.num_clusters = num_clusters

        # creat nets
        conv_k = conv_encoder(in_channels=in_channels)
        self.encoder_k = nn.Sequential(conv_k,
                                       nn.AdaptiveAvgPool2d(1),
                                       nn.Flatten(1),
                                       nn.Linear(512, 512),
                                       nn.ReLU(),
                                       nn.Linear(512, fea_dim))

        conv_qs = conv_encoder(in_channels=in_channels)
        self.encoder_q = nn.Sequential(conv_qs,
                                       nn.AdaptiveAvgPool2d(1),
                                       nn.Flatten(1),
                                       nn.Linear(512, 512),
                                       nn.ReLU(),
                                       nn.Linear(512, fea_dim))
        self.encoder_s = nn.Sequential(conv_qs,  # Share the common conv_encoder with net_q.
                                       nn.AdaptiveAvgPool2d(1),
                                       nn.Flatten(1),
                                       nn.Linear(512, 512),
                                       nn.ReLU(),
                                       nn.Linear(512, num_clusters))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        if freeze_encoder_s:
            self.set_grad(self.encoder_s, False)

        # create the queue
        self.register_buffer("queue", torch.randn(fea_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def set_grad(self, modules, TF):
        for param in modules.parameters():
            param.requires_grad = TF

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        # self.queue[:, ptr:ptr + batch_size] = keys.T
        # print(keys.shape)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(1, 0)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward_sim(self, im_q, im_k, update_k=True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            if update_k:
                self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    def forward_sim2sem(self, im_qs, im_k, return_cls, return_sim, return_semantic_sim):
        # Update k_net, and compute sim features of im_k.
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k_feas = self.encoder_k(im_k)  # keys: NxC
            k_feas = nn.functional.normalize(k_feas, dim=1)

        # Compute the cluster probabilities of im_qs.
        with torch.no_grad():
            score_s = self.encoder_s(im_qs)
            prob_s = torch.softmax(score_s, dim=1)

        # Compute the cluster centers.
        _, idx_max = torch.sort(prob_s, dim=0, descending=True)
        idx_max = idx_max[0:self.center_k, :]

        centers = []
        for c in range(idx_max.shape[1]):
            centers.append(k_feas[idx_max[:, c], :].mean(dim=0).unsqueeze(dim=0))

        centers = torch.cat(centers, dim=0)

        # Select balanced training samples.
        num_per_cluster = int(im_qs.shape[0] * self.sr) // self.num_clusters
        dis = torch.einsum('cd,nd->cn', [centers, k_feas])
        idx_select = torch.argsort(dis, dim=1)[:, ::-1][:, 0:num_per_cluster].flatten()

        im_qs_select = im_qs[idx_select, ...]
        im_k_select = im_k[idx_select, ...]


        output_all = []
        if return_cls:
            # compute the output of encoder_s on the select_images.
            out_s = self.encoder_s(im_qs_select)
            labels_s = torch.arange(0, self.num_clusters).unsqueeze(dim=1).repeat(1, num_per_cluster).flatten()
            output_all.append(out_s)
            output_all.append(labels_s)

        if return_sim:
            # Compute the output and labels of encoder_q on the select_images.
            out_q, labels_q = self.forward_sim(im_qs_select, im_k_select, update_k=False)
            output_all.append(out_q)
            output_all.append(labels_q)

        if return_semantic_sim:
            # Compute the label features of im_qs_select and im_k_select.
            with torch.no_grad():  # no gradient to keys
                # shuffle for making use of BN
                sccore_k = self.encoder_s(im_k_select)  # keys: NxC
                prob_k = nn.functional.softmax(sccore_k, dim=1)
                prob_norm_k = nn.functional.normalize(prob_k, dim=1)

            prob_s = nn.functional.softmax(out_s, dim=1)
            prob_norm_s = nn.functional.normalize(prob_s, dim=1)
            output_all.append(prob_norm_s, prob_norm_k)

        return output_all

    def forward_clustering(self, img):
        with torch.no_grad():
            scores = self.encoder_s(img)
            probs = nn.functional.softmax(scores, dim=1)

        cluster_labels = probs.argmax(dim=1)
        return probs, cluster_labels

    def forward(self, img1, img2=None, forward_type=None, return_cls=True, return_sim=True, return_semantic_sim=False):
        if forward_type == "sim":
            return self.forward_sim(img1, img2)
        elif forward_type == "sim2sem":
            return self.forward_sim2sem(img1, img2, return_cls, return_sim, return_semantic_sim)
        elif forward_type == "clustering":
            return self.forward_clustering(img1)
        else:
            raise TypeError


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
