import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight.torchlight import import_class
import random
from .skeletonAdaIN import AdaIN
def loss_fn(p, z,type='l1'):
    z = z.detach()
    if type=='l1':
        loss = nn.L1Loss()
        return 5* loss(p,z)
    return -7 * nn.functional.cosine_similarity(p, z, dim=-1).mean()

class AimCLR(nn.Module):
    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5,dropout_graph=0.1,add_graph=0.05,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, transformer=False, rep_ratio=0,doublequeue=False,**kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain
        if pretrain:
            self.register_parameter('mask_param', nn.Parameter(torch.zeros((1,))) )
        if not self.pretrain:
            if transformer:
                self.encoder_q = base_encoder(num_class= num_class, **kwargs)
            else:
                self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          dropout_graph=dropout_graph,add_graph=add_graph,
                                          **kwargs)
        else:
            self.K = queue_size
            self.m = momentum
            self.T = Temperature
            print('dropout_graph=',dropout_graph,'add_graph=',add_graph)
            if transformer:
                self.encoder_q = base_encoder(num_class= feature_dim, **kwargs)
                self.encoder_k = base_encoder(num_class= feature_dim, **kwargs)
            else:
                self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                            hidden_dim=hidden_dim, num_class=feature_dim,
                                            dropout=dropout, graph_args=graph_args,
                                            edge_importance_weighting=edge_importance_weighting,
                                            dropout_graph=dropout_graph,add_graph=add_graph,
                                            **kwargs)
                self.encoder_k = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                            hidden_dim=hidden_dim, num_class=feature_dim,
                                            dropout=dropout, graph_args=graph_args,
                                            edge_importance_weighting=edge_importance_weighting,
                                            dropout_graph=dropout_graph,add_graph=add_graph,
                                            **kwargs)

            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_k.fc)
            
            #self.last_fc1 = nn.Sequential(nn.ReLU(),
            #                              nn.Linear(feature_dim, feature_dim))
            #self.last_fc2 = nn.Sequential(nn.ReLU(),
            #                              nn.Linear(feature_dim, feature_dim))
            #self.last_fc3 = nn.Sequential(nn.ReLU(),
            #                              nn.Linear(feature_dim, feature_dim))
            self.ratio = rep_ratio# the ratio of (selected normal)/(total replace)
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

            # create the queue
            if doublequeue:
                self.register_buffer("queue_str", torch.randn(feature_dim, queue_size))
                self.queue_str = F.normalize(self.queue_str, dim=0)

            self.register_buffer("queue", torch.randn(feature_dim, queue_size))
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        gpu_index = keys.device.index
        self.queue[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T
    
    @torch.no_grad()
    def _dequeue_and_enqueue_str(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        gpu_index = keys.device.index
        self.queue_str[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T
    
    @torch.no_grad()
    def _dequeue_and_enqueue_wstrong(self, keys_nor,keys_str):
        if self.ratio==0.0:
            print('Warning: random enqueue is used, but replace ratio is 0!')
        batch_size = keys_nor.shape[0]
        ptr = int(self.queue_ptr)
        gpu_index = keys_nor.device.index
        nor_idx = random.sample(range(batch_size), int(batch_size*self.ratio))
        new_key = keys_str
        new_key[nor_idx] = keys_nor[nor_idx]
        self.queue[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = new_key.T
    
    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0  # for simplicity
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K

    def forward(self, im_q_extreme, im_q, im_k=None, nnm=False, topk=1,KNN=False,return_feat=False):
        """
        random dropout and add edges for adj graph
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """
        if nnm:
            return self.nearest_neighbors_mining(im_q, im_k, im_q_extreme, topk)

        if not self.pretrain:
            return self.encoder_q(im_q,KNN=KNN,return_feat=return_feat)
    def forward_daGraph(self, im_q_extreme, im_q, im_k=None, nnm=False, topk=1,mask=None):
        """
        random dropout and add edges for adj graph
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """
        if nnm:
            return self.nearest_neighbors_mining(im_q, im_k, im_q_extreme, topk)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        if mask==None:
            q_extreme, q_extreme_drop = self.encoder_q(im_q_extreme, drop_graph=True)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        # Compute logits of normally augmented query using Einstein sum
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

        # Compute logits_e of extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        # apply temperature
        logits_e /= self.T
        logits_e = torch.softmax(logits_e, dim=1)

        # Compute logits_ed of dropped extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)
        # apply temperature
        logits_ed /= self.T
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()
                
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, logits_e, logits_ed, labels_ddm
    def forward_cossim(self, im_q_extreme, im_q, im_k=None, nnm=False, topk=1,mask=None):
        """
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """

        if nnm:
            return self.nearest_neighbors_mining_cossim(im_q, im_k, im_q_extreme, topk)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        if mask==None:
            q_extreme, q_extreme_drop = self.encoder_q(im_q_extreme, drop_graph=True)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        cos_loss = loss_fn(q_extreme,k)/2
        cos_loss += loss_fn(q_extreme_drop,k)/2


        # Compute logits of normally augmented query using Einstein sum
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

        return logits, labels, cos_loss
    def forward_cossim_hierarchical(self, im_q_extreme, im_q, im_k=None, nnm=False, topk=1,mask=None):
        """
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """

        if nnm:
            return self.nearest_neighbors_mining_cossim_hierarchical(im_q, im_k, im_q_extreme, topk)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        if mask==None:
            q_extreme, q_extreme_drop = self.encoder_q(im_q_extreme, drop_graph=True)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            k2 = self.encoder_k(im_q_extreme)  # keys: NxC
            k2 = F.normalize(k2, dim=1)

        cos_loss = loss_fn(q_extreme,k)/2
        cos_loss += loss_fn(q_extreme_drop,k2)/2


        # Compute logits of normally augmented query using Einstein sum
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

        return logits, labels, cos_loss
    def forward_baseline(self, im_q_extreme, im_q, im_k=None, nnm=False, topk=1,mask=None):
        """
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """
        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature


        # Normalize the feature
        q = F.normalize(q, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        # Compute logits of normally augmented query using Einstein sum
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
    def forward_mutalddm(self, im_q_extreme, im_q, im_k=None, nnm=False, topk=1,mask=None):
        """
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """
        if mask != None:
            im_mask = im_q.permute(0,2,4,3,1)#NTMVC 
            im_mask = im_mask*mask
            im_mask[mask==0] = self.mask_param
            im_mask = im_mask.permute(0,4,1,3,2)#NCTVM
        

        if nnm:
            return self.nearest_neighbors_mining_mutalddm(im_q, im_k, im_q_extreme, topk)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        if mask==None:
            q_extreme, q_extreme_drop = self.encoder_q(im_q_extreme, drop_graph=True)  # NxC
        else:
            q_extreme, q_extreme_drop = self.encoder_q(im_mask, drop_graph=True)

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        # Compute logits of normally augmented query using Einstein sum
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

        # Compute logits_e of extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        # apply temperature
        logits_e /= self.T
        logits_e_ddm = logits_e
        logits_e = torch.softmax(logits_e, dim=1)

        # Compute logits_ed of dropped extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)
        # apply temperature
        logits_ed /= self.T
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()
        
        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, logits_e, logits_ed, labels_ddm, labels_ddm2
    def forward_mutalddm_multinm(self, im_q_extreme, im_q, im_k=None, nnm=False, topk=1,mask=None):
        """
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """
        if mask != None:
            im_mask = im_q.permute(0,2,4,3,1)#NTMVC 
            im_mask = im_mask*mask
            im_mask[mask==0] = self.mask_param
            im_mask = im_mask.permute(0,4,1,3,2)#NCTVM
        

        if nnm:
            return self.nearest_neighbors_mining_mutalddm_multinm(im_q, im_k, im_q_extreme, topk)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        if mask==None:
            q_extreme, q_extreme_drop = self.encoder_q(im_q_extreme, drop_graph=True)  # NxC
        else:
            q_extreme, q_extreme_drop = self.encoder_q(im_mask, drop_graph=True)

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        # Compute logits of normally augmented query using Einstein sum
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

        # Compute logits_e of extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        # apply temperature
        logits_e /= self.T
        logits_e_ddm = logits_e
        logits_e = torch.softmax(logits_e, dim=1)

        # Compute logits_ed of dropped extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)
        # apply temperature
        logits_ed /= self.T
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()
        
        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, logits_e, logits_ed, labels_ddm, labels_ddm2
    def forward_mutalddm_cosnnm(self, im_q_extreme, im_q, im_k=None, nnm=False, topk=1,mask=None):
        """
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """
        if mask != None:
            im_mask = im_q.permute(0,2,4,3,1)#NTMVC 
            im_mask = im_mask*mask
            im_mask[mask==0] = self.mask_param
            im_mask = im_mask.permute(0,4,1,3,2)#NCTVM
        

        if nnm:
            return self.nearest_neighbors_mining_mutalddm_cosnnm(im_q, im_k, im_q_extreme, topk)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        if mask==None:
            q_extreme, q_extreme_drop = self.encoder_q(im_q_extreme, drop_graph=True)  # NxC
        else:
            q_extreme, q_extreme_drop = self.encoder_q(im_mask, drop_graph=True)

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        # Compute logits of normally augmented query using Einstein sum
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

        # Compute logits_e of extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        # apply temperature
        logits_e /= self.T
        logits_e_ddm = logits_e
        logits_e = torch.softmax(logits_e, dim=1)

        # Compute logits_ed of dropped extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)
        # apply temperature
        logits_ed /= self.T
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()
        
        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, logits_e, logits_ed, labels_ddm, labels_ddm2
    def forward_mutalddm_with4p(self, im_q_extreme1,im_q_extreme2, im_q, im_k=None, nnm=False, topk=1,mask=None):
        """
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """
        
        if nnm:
            return self.nearest_neighbors_mining_mutalddm_with4p(im_q, im_k, im_q_extreme1,im_q_extreme2, topk)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        if mask==None:
            q_extreme = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC
            q_extreme_drop = self.encoder_q(im_q_extreme2, drop_graph=True,return1d=True)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        # Compute logits of normally augmented query using Einstein sum
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

        # Compute logits_e of extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        # apply temperature
        logits_e /= self.T
        logits_e_ddm = logits_e
        logits_e = torch.softmax(logits_e, dim=1)

        # Compute logits_ed of dropped extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)
        # apply temperature
        logits_ed /= self.T
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()
        
        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, logits_e, logits_ed, labels_ddm, labels_ddm2
    def forward_abl_modeltrans(self, im_q, im_k=None, mask=None):
        """
        Abl exp for skeletonclr
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """

        if mask != None:
            im_q = im_q.permute(0,2,4,3,1)#NTMVC 
            im_q = im_q*mask
            im_q[mask==0] = self.mask_param
            im_q = im_q.permute(0,4,1,3,2)#NCTVM

        if not self.pretrain:
            return self.encoder_q(im_q)

        # compute query features
        q = self.encoder_q(im_q,drop_graph=True,return1d=True)  # queries: NxC
        q = F.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

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
    def forward_abl_adain(self, im_q, im_k=None, mask=None):
        """
        Abl exp for skeletonclr
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """

        if mask != None:
            im_q = im_q.permute(0,2,4,3,1)#NTMVC 
            im_q = im_q*mask
            im_q[mask==0] = self.mask_param
            im_q = im_q.permute(0,4,1,3,2)#NCTVM

        if not self.pretrain:
            return self.encoder_q(im_q)
        N = im_q.shape[0]
        idx = torch.arange(N)
        n1 = torch.randint(1, N - 1, (1,))
        n2 = torch.randint(1, N - 1, (1,))
        randidx = (idx + n1) % N
        randidx2 = (idx + n2) % N
        im_q = AdaIN(im_q, im_q[randidx])
        im_k = AdaIN(im_k, im_k[randidx2])
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = F.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

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
    def forward_mutalddm_with4p_labelsmooth(self, im_q_extreme1,im_q_extreme2, im_q, im_k=None, nnm=False, topk=1,mask=None):
        """
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """
        
        if nnm:
            return self.nearest_neighbors_mining_mutalddm_with4p(im_q, im_k, im_q_extreme1,im_q_extreme2, topk)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        if mask==None:
            q_extreme = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC
            q_extreme_drop = self.encoder_q(im_q_extreme2, drop_graph=True,return1d=True)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        # Compute logits of normally augmented query using Einstein sum
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.T
        # labels: positive key indicators
        logits_q = torch.zeros(logits.shape[1],dtype=torch.float).cuda()
        alpha = 0.1
        logits[:] = alpha/(logits.shape[1])
        logits_q[0] += (1-alpha)

        #labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # Compute logits_e of extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        # apply temperature
        logits_e /= self.T
        logits_e_ddm = logits_e
        logits_e = torch.softmax(logits_e, dim=1)

        # Compute logits_ed of dropped extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)
        # apply temperature
        logits_ed /= self.T
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()
        
        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, logits_q, logits_e, logits_ed, labels_ddm, labels_ddm2
    def forward_mutalddm_with4p_adain(self, im_q_extreme1,im_q_extreme2, im_q, im_k=None, nnm=False, topk=1,mask=None, randenque=False):
        """
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """
        
        if nnm:
            return self.nearest_neighbors_mining_mutalddm_with4p_adain(im_q, im_k, im_q_extreme1,im_q_extreme2, topk, randenque=randenque)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        if mask==None:
            q_extreme = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC
            N = q.shape[0]
            idx = torch.arange(N)
            n = torch.randint(1, N - 1, (1,))
            randidx = (idx + n) % N
            im_q_extreme2 = AdaIN(im_q_extreme2, im_q_extreme2[randidx])
            q_extreme_drop = self.encoder_q(im_q_extreme2, drop_graph=False)  # NxC
        
        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            if randenque:
                idx_k = torch.arange(N)
                n_k = torch.randint(1, N - 1, (1,))
                randidx_k = (idx_k + n_k) % N
                im_q_extreme2_k = AdaIN(im_q_extreme2, im_q_extreme2[randidx_k])
                k_str = self.encoder_k(im_q_extreme2_k) 
                k_str = F.normalize(k_str, dim=1)

        # Compute logits of normally augmented query using Einstein sum
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

        # Compute logits_e of extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        # apply temperature
        logits_e /= self.T
        logits_e_ddm = logits_e
        logits_e = torch.softmax(logits_e, dim=1)

        # Compute logits_ed of dropped extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)
        # apply temperature
        logits_ed /= self.T
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()
        
        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # dequeue and enqueue
        if randenque:
            self._dequeue_and_enqueue_wstrong(k, k_str)
        else:
            self._dequeue_and_enqueue(k)

        return logits, labels, logits_e, logits_ed, labels_ddm, labels_ddm2
    def forward_mutalddm_with4p_adain_wrandenque(self, im_q_extreme1,im_q_extreme2, im_q, im_k=None, im_k_str=None, nnm=False, topk=1,mask=None, randenque=False):
        """
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """
        
        if nnm:
            return self.nearest_neighbors_mining_mutalddm_with4p_adain_wrandenque(im_q, im_k, im_q_extreme1,im_q_extreme2,im_k_str, topk, randenque=randenque)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        if mask==None:
            q_extreme = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC
            N = q.shape[0]
            idx = torch.arange(N)
            n = torch.randint(1, N - 1, (1,))
            randidx = (idx + n) % N
            im_q_extreme2 = AdaIN(im_q_extreme2, im_q_extreme2[randidx])
            q_extreme_drop = self.encoder_q(im_q_extreme2, drop_graph=False)  # NxC
        
        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            if randenque:
                k_str = self.encoder_k(im_k_str)  # keys: NxC
                k_str = F.normalize(k_str, dim=1)


        # Compute logits of normally augmented query using Einstein sum
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

        # Compute logits_e of extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        # apply temperature
        logits_e /= self.T
        logits_e_ddm = logits_e
        logits_e = torch.softmax(logits_e, dim=1)

        # Compute logits_ed of dropped extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)
        # apply temperature
        logits_ed /= self.T
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()
        
        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # dequeue and enqueue
        if randenque:
            self._dequeue_and_enqueue_wstrong(k, k_str)
        else:
            self._dequeue_and_enqueue(k)

        return logits, labels, logits_e, logits_ed, labels_ddm, labels_ddm2
    def forward_mutalddm_with4p_wmask(self, im_q_extreme1,im_q_extreme2, im_q, im_k=None, nnm=False, topk=1,mask=None):
        """
        在第四路再加入mask增强
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """
        if mask != None:
            im_q_extreme2 = im_q_extreme2.permute(0,2,4,3,1)#NTMVC 
            im_q_extreme2 = im_q_extreme2*mask
            im_q_extreme2[mask==0] = self.mask_param
            im_q_extreme2 = im_q_extreme2.permute(0,4,1,3,2)#NCTVM
        
        if nnm:
            return self.nearest_neighbors_mining_mutalddm_with4p(im_q, im_k, im_q_extreme1,im_q_extreme2, topk)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        if 1:
            q_extreme = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC
            q_extreme_drop = self.encoder_q(im_q_extreme2, drop_graph=True,return1d=True)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        # Compute logits of normally augmented query using Einstein sum
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

        # Compute logits_e of extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        # apply temperature
        logits_e /= self.T
        logits_e_ddm = logits_e
        logits_e = torch.softmax(logits_e, dim=1)

        # Compute logits_ed of dropped extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)
        # apply temperature
        logits_ed /= self.T
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()
        
        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, logits_e, logits_ed, labels_ddm, labels_ddm2
    def forward_mutalddm_with4p_ablmask(self, im_q_extreme1,im_q_extreme2, im_q, im_k=None, im_k_str=None, nnm=False, topk=1,mask=None):
        """
        第四路只有mask
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """
        if mask != None:
            im_q_extreme2 = im_q_extreme2.permute(0,2,4,3,1)#NTMVC 
            im_q_extreme2 = im_q_extreme2*mask
            im_q_extreme2[mask==0] = self.mask_param
            im_q_extreme2 = im_q_extreme2.permute(0,4,1,3,2)#NCTVM
        if nnm:
            return self.nearest_neighbors_mining_mutalddm_with4p_ablmask(im_q, im_k, im_q_extreme1,im_q_extreme2, topk, im_k_str=im_k_str)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        if mask!=None:
            q_extreme = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC
            q_extreme_drop = self.encoder_q(im_q_extreme2, drop_graph=False)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            if im_k_str!=None:
                k_str = self.encoder_k(im_k_str)  # keys: NxC
                k_str = F.normalize(k_str, dim=1)                

        # Compute logits of normally augmented query using Einstein sum
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

        # Compute logits_e of extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        # apply temperature
        logits_e /= self.T
        logits_e_ddm = logits_e
        logits_e = torch.softmax(logits_e, dim=1)

        # Compute logits_ed of dropped extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)
        # apply temperature
        logits_ed /= self.T
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()
        
        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # dequeue and enqueue
        if im_k_str!=None:
            self._dequeue_and_enqueue_wstrong(k,k_str)
        else:
            self._dequeue_and_enqueue(k)

        return logits, labels, logits_e, logits_ed, labels_ddm, labels_ddm2
    
    def forward_mutalddm_with4p_ablmask_cpm(self, im_q_extreme1,im_q_extreme2, im_q, im_k=None, im_k_str=None, nnm=False, topk=1,mask=None):
        """
        第四路只有mask
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """
        t_prime = 0.05 #temperature for the stopgrad
        if mask != None:
            im_q_extreme2 = im_q_extreme2.permute(0,2,4,3,1)#NTMVC 
            im_q_extreme2 = im_q_extreme2*mask
            im_q_extreme2[mask==0] = self.mask_param
            im_q_extreme2 = im_q_extreme2.permute(0,4,1,3,2)#NCTVM
        if nnm:
            return self.nearest_neighbors_mining_mutalddm_with4p_ablmask(im_q, im_k, im_q_extreme1,im_q_extreme2, topk, im_k_str=im_k_str)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        if mask!=None:
            q_extreme = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC
            q_extreme_drop = self.encoder_q(im_q_extreme2, drop_graph=False)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            if im_k_str!=None:
                k_str = self.encoder_k(im_k_str)  # keys: NxC
                k_str = F.normalize(k_str, dim=1)                

        # Compute logits of normally augmented query using Einstein sum
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits_label = logits / t_prime
        logits /= self.T
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # Compute logits_e of extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        # apply temperature
        logits_e_ddm = logits_e / t_prime
        logits_e /= self.T
        logits_e = torch.softmax(logits_e, dim=1)

        # Compute logits_ed of dropped extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)
        # apply temperature
        logits_ed /= self.T
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_e, topk, dim=1)

        labels_ddm = logits_label.clone().detach()
        sum_labels_ddm = torch.sum(torch.exp(labels_ddm),dim=1,keepdim=True) #n,1
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm.scatter_(1, topkdix, torch.exp(1./t_prime)/sum_labels_ddm)
        labels_ddm = labels_ddm.detach()
        
        labels_ddm2 = logits_e_ddm.clone().detach()
        sum_labels_ddm2 = torch.sum(torch.exp(labels_ddm2),dim=1,keepdim=True) #n,1
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2.scatter_(1, topkdix_e, torch.exp(1./t_prime)/sum_labels_ddm2)
        labels_ddm2 = labels_ddm2.detach()

        
        # dequeue and enqueue
        if im_k_str!=None:
            self._dequeue_and_enqueue_wstrong(k,k_str)
        else:
            self._dequeue_and_enqueue(k)

        return logits, labels, logits_e, logits_ed, labels_ddm, labels_ddm2

    def forward_mutalddm_with4p_ablmask_disorder(self, im_q_extreme1,im_q_extreme2, im_q, im_k=None, im_k_str=None, nnm=False, topk=1,mask=None):
        """
        消融实验 分层顺序 先ba+na，再ba na
        #im_q:origi
        #im_k:origin
        #im_q_ex1: ba
        #im_q_ex2: ba+na
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """
        if mask != None:
            m1,m2,m3,m4 = mask
            im_q_extreme2 = im_q_extreme2.permute(0,2,4,3,1)#NTMVC 
            im_q_extreme2 = im_q_extreme2*m4
            im_q_extreme2[m4==0] = self.mask_param
            im_q_extreme2 = im_q_extreme2.permute(0,4,1,3,2)#NCTVM

            im_q_extreme1 = im_q_extreme1.permute(0,2,4,3,1)#NTMVC 
            im_q_extreme1 = im_q_extreme1*m3
            im_q_extreme1[m3==0] = self.mask_param
            im_q_extreme1 = im_q_extreme1.permute(0,4,1,3,2)#NCTVM

            im_q = im_q.permute(0,2,4,3,1)#NTMVC 
            im_q = im_q*m2
            im_q[m2==0] = self.mask_param
            im_q = im_q.permute(0,4,1,3,2)#NCTVM

            im_k = im_k.permute(0,2,4,3,1)#NTMVC 
            im_k = im_k*m1
            im_k[m1==0] = self.mask_param
            im_k = im_k.permute(0,4,1,3,2)#NCTVM

        if nnm:
            return self.nearest_neighbors_mining_mutalddm_with4p_ablmask(im_q, im_k, im_q_extreme1,im_q_extreme2, topk, im_k_str=im_k_str)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        if mask!=None:
            q_extreme = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC
            q_extreme_drop = self.encoder_q(im_q_extreme2, drop_graph=False)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            if im_k_str!=None:
                k_str = self.encoder_k(im_k_str)  # keys: NxC
                k_str = F.normalize(k_str, dim=1)                

        # Compute logits of normally augmented query using Einstein sum
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

        # Compute logits_e of extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        # apply temperature
        logits_e /= self.T
        logits_e_ddm = logits_e
        logits_e = torch.softmax(logits_e, dim=1)

        # Compute logits_ed of dropped extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)
        # apply temperature
        logits_ed /= self.T
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()
        
        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # dequeue and enqueue
        if im_k_str!=None:
            self._dequeue_and_enqueue_wstrong(k,k_str)
        else:
            self._dequeue_and_enqueue(k)

        return logits, labels, logits_e, logits_ed, labels_ddm, labels_ddm2
    def forward_mutalddm_with4p_ablmask_cos(self, im_q_extreme1,im_q_extreme2, im_q, im_k=None, im_k_str=None, nnm=False, topk=1,mask=None):
        """
        第四路只有mask
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """
        if mask != None:
            im_q_extreme2 = im_q_extreme2.permute(0,2,4,3,1)#NTMVC 
            im_q_extreme2 = im_q_extreme2*mask
            im_q_extreme2[mask==0] = self.mask_param
            im_q_extreme2 = im_q_extreme2.permute(0,4,1,3,2)#NCTVM
        if nnm:
            return self.nearest_neighbors_mining_mutalddm_with4p_ablmask_cos(im_q, im_k, im_q_extreme1,im_q_extreme2, topk, im_k_str=im_k_str)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        if mask!=None:
            q_extreme = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC
            q_extreme_drop = self.encoder_q(im_q_extreme2, drop_graph=False)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            if im_k_str!=None:
                k_str = self.encoder_k(im_k_str)  # keys: NxC
                k_str = F.normalize(k_str, dim=1)                

        # Compute logits of normally augmented query using Einstein sum
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

        cos_loss = loss_fn(q_extreme, q.detach())
        cos_loss += loss_fn(q_extreme_drop, q_extreme.detach())

        # dequeue and enqueue
        if im_k_str!=None:
            self._dequeue_and_enqueue_wstrong(k,k_str)
        else:
            self._dequeue_and_enqueue(k)

        return logits, labels,cos_loss
    def forward_mutalddm_with4p_ablmask_topology(self, im_q_extreme1,im_q_extreme2, im_q, im_k=None, im_k_str=None, nnm=False, topk=1,mask=None):
        """
        第四路只有mask
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """
        if mask != None:
            im_q_extreme2 = im_q_extreme2.permute(0,2,4,3,1)#NTMVC 
            im_q_extreme2 = im_q_extreme2*mask
            im_q_extreme2[mask==0] = self.mask_param[(mask==0)[:-2]]
            im_q_extreme2 = im_q_extreme2.permute(0,4,1,3,2)#NCTVM
        if nnm:
            return self.nearest_neighbors_mining_mutalddm_with4p_ablmask_topology(im_q, im_k, im_q_extreme1,im_q_extreme2, topk, im_k_str=im_k_str)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        if mask!=None:
            q_extreme = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC
            q_extreme_drop = self.encoder_q(im_q_extreme2, drop_graph=False, mask_pool=True, mask=mask)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            if im_k_str!=None:
                k_str = self.encoder_k(im_k_str)  # keys: NxC
                k_str = F.normalize(k_str, dim=1)                

        # Compute logits of normally augmented query using Einstein sum
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

        # Compute logits_e of extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        # apply temperature
        logits_e /= self.T
        logits_e_ddm = logits_e
        logits_e = torch.softmax(logits_e, dim=1)

        # Compute logits_ed of dropped extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)
        # apply temperature
        logits_ed /= self.T
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()
        
        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # dequeue and enqueue
        if im_k_str!=None:
            self._dequeue_and_enqueue_wstrong(k,k_str)
        else:
            self._dequeue_and_enqueue(k)

        return logits, labels, logits_e, logits_ed, labels_ddm, labels_ddm2
    def forward_mutalddm_abl_onlystr(self, im_q_extreme1, im_q, im_k=None, nnm=False, topk=1,mask=None):
        """
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """
        
        if nnm:
            return self.nearest_neighbors_mining_mutalddm_abl_onlystr(im_q, im_k, im_q_extreme1, topk)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        if mask==None:
            q_extreme = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        # Compute logits of normally augmented query using Einstein sum
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

        # Compute logits_e of extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        # apply temperature
        logits_e /= self.T
        logits_e_ddm = logits_e
        logits_e = torch.softmax(logits_e, dim=1)
  
        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, logits_e, labels_ddm
    def forward_mutalddm_abl_disorder_3p(self, im_q_extreme1, im_q, im_k=None, nnm=False, topk=1,mask=None):
        """
        只有2路 ba+na,mask
        im_q im_k: ba+na
        im_q_extre: ba+na+mask
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """
        if mask != None:
            im_q_extreme1 = im_q_extreme1.permute(0,2,4,3,1)#NTMVC 
            im_q_extreme1 = im_q_extreme1*mask
            im_q_extreme1[mask==0] = self.mask_param
            im_q_extreme1 = im_q_extreme1.permute(0,4,1,3,2)#NCTVM

        if nnm:
            return self.nearest_neighbors_mining_mutalddm_abl_onlystr(im_q, im_k, im_q_extreme1, topk)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        
        q_extreme = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        # Compute logits of normally augmented query using Einstein sum
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

        # Compute logits_e of extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        # apply temperature
        logits_e /= self.T
        logits_e_ddm = logits_e
        logits_e = torch.softmax(logits_e, dim=1)
  
        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, logits_e, labels_ddm
    def forward_mutalddm_with4p_randenque(self, im_q_extreme1,im_q_extreme2, im_q, im_k=None, im_k_str=None, nnm=False, topk=1,mask=None):
        """
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """
        
        if nnm:
            return self.nearest_neighbors_mining_mutalddm_with4p_randenque(im_q, im_k, im_k_str, im_q_extreme1,im_q_extreme2, topk)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        if mask==None:
            q_extreme = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC
            q_extreme_drop = self.encoder_q(im_q_extreme2, drop_graph=True,return1d=True)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            k_str = self.encoder_k(im_k_str)
            k_str = F.normalize(k_str, dim=1)

        # Compute logits of normally augmented query using Einstein sum
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

        # Compute logits_e of extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        # apply temperature
        logits_e /= self.T
        logits_e_ddm = logits_e
        logits_e = torch.softmax(logits_e, dim=1)

        # Compute logits_ed of dropped extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)
        # apply temperature
        logits_ed /= self.T
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()
        
        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # dequeue and enqueue
        self._dequeue_and_enqueue_wstrong(k,k_str)

        return logits, labels, logits_e, logits_ed, labels_ddm, labels_ddm2
    def forward_mutalddm_with4p_doubleque(self, im_q_extreme1,im_q_extreme2, im_q, im_k=None, im_k_str=None, nnm=False, topk=1,mask=None):
        """
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """
        
        if nnm:
            return self.nearest_neighbors_mining_mutalddm_with4p_doubleque(im_q, im_k, im_k_str, im_q_extreme1,im_q_extreme2, topk)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        if mask==None:
            q_extreme = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC
            q_extreme_drop = self.encoder_q(im_q_extreme2, drop_graph=True,return1d=True)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            k_str = self.encoder_k(im_k_str)
            k_str = F.normalize(k_str, dim=1)

        # Compute logits of normally augmented query using Einstein sum
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

        # Compute logits_e of extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        # apply temperature
        logits_e /= self.T
        logits_e = torch.softmax(logits_e, dim=1)

        #compute the target for the modeltrans branch
        l_pos_e_tar = torch.einsum('nc,nc->n', [q_extreme, k_str]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_e_tar = torch.einsum('nc,ck->nk', [q_extreme, self.queue_str.clone().detach()])
        # logits: Nx(1+K)
        logits_e_tar = torch.cat([l_pos_e_tar, l_neg_e_tar], dim=1)
        # apply temperature
        logits_e_tar /= self.T
        logits_e_ddm = logits_e_tar


        # Compute logits_ed of dropped extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k_str]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue_str.clone().detach()])
        # logits: Nx(1+K)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)
        # apply temperature
        logits_ed /= self.T
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()
        
        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue_str(k_str)

        return logits, labels, logits_e, logits_ed, labels_ddm, labels_ddm2
    def forward_mutalddm_with5p_randenque(self, im_q_extreme1,im_q_extreme2, im_q, im_k=None, im_k_str=None, nnm=False, topk=1,mask=None):
        """
        normal+strong+model trans+AdaIN
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """
        
        if nnm:
            return self.nearest_neighbors_mining_mutalddm_with5p_randenque(im_q, im_k, im_k_str, im_q_extreme1,im_q_extreme2, topk)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        if mask==None:
            q_extreme = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC
            q_extreme_drop = self.encoder_q(im_q_extreme2, drop_graph=True,return1d=True)  # NxC
            N = q.shape[0]
            idx = torch.arange(N)
            n = torch.randint(1, N - 1, (1,))
            randidx = (idx + n) % N
            im_q_extreme3 = AdaIN(im_k_str, im_k_str[randidx],dim='V')
            q_extreme_drop3 = self.encoder_q(im_q_extreme3, drop_graph=True,return1d=True)  # NxC


        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)
        q_extreme_drop3 = F.normalize(q_extreme_drop3, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            k_str = self.encoder_k(im_k_str)
            k_str = F.normalize(k_str, dim=1)

        # Compute logits of normally augmented query using Einstein sum
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

        # Compute logits_e of extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        # apply temperature
        logits_e /= self.T
        logits_e_ddm = logits_e
        logits_e = torch.softmax(logits_e, dim=1)

        # Compute logits_ed of dropped extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)
        # apply temperature
        logits_ed /= self.T
        logits_ed_ddm = logits_ed
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Compute logits_edd of modeltrans+AdaIN augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_edd = torch.einsum('nc,nc->n', [q_extreme_drop3, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_edd = torch.einsum('nc,ck->nk', [q_extreme_drop3, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_edd = torch.cat([l_pos_edd, l_neg_edd], dim=1)
        # apply temperature
        logits_edd /= self.T
        logits_edd = torch.softmax(logits_edd, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()
        
        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        labels_ddm3 = logits_ed_ddm.clone().detach()
        labels_ddm3 = torch.softmax(labels_ddm3, dim=1)
        labels_ddm3 = labels_ddm3.detach()
        # dequeue and enqueue
        self._dequeue_and_enqueue_wstrong(k,k_str)

        return logits, labels, logits_e, logits_ed, logits_edd, labels_ddm, labels_ddm2, labels_ddm3
    def forward_mutalddm_with4p_mlp2(self, im_q_extreme1,im_q_extreme2, im_q, im_k=None, nnm=False, topk=1,mask=None):
        """
        only add a mlp at the last of each branch
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """
        
        if nnm:
            return self.nearest_neighbors_mining_mutalddm_with4p(im_q, im_k, im_q_extreme1,im_q_extreme2, topk)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        if mask==None:
            q_extreme = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC
            #q_extreme = self.last_fc2(q_extreme)
            q_extreme_drop = self.encoder_q(im_q_extreme2, drop_graph=True,return1d=True)  # NxC
            #q_extreme_drop = self.last_fc1(q_extreme_drop)

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        # Compute logits of normally augmented query using Einstein sum
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

        # Compute logits_e of extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        # apply temperature
        logits_e /= self.T
        logits_e_ddm = logits_e
        logits_e = torch.softmax(logits_e, dim=1)

        # Compute logits_ed of dropped extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)
        # apply temperature
        logits_ed /= self.T
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()
        
        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, logits_e, logits_ed, labels_ddm, labels_ddm2
    def forward_mutalddm_with4p_mlp(self, im_q_extreme1,im_q_extreme2, im_q, im_k=None, nnm=False, topk=1,mask=None):
        """
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """
        
        if nnm:
            return self.nearest_neighbors_mining_mutalddm_with4p_mlp(im_q, im_k, im_q_extreme1,im_q_extreme2, topk)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        if mask==None:
            q_extreme_lab = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC
            q_extreme = self.last_fc2(q_extreme_lab)

            q_extreme_drop = self.encoder_q(im_q_extreme2, drop_graph=True,return1d=True)  # NxC
            q_extreme_drop = self.last_fc1(q_extreme_drop)

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_lab = F.normalize(q_extreme_lab, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        # Compute logits of normally augmented query using Einstein sum
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

        # Compute logits_e of extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        # apply temperature
        logits_e /= self.T
        #logits_e_ddm = logits_e
        logits_e = torch.softmax(logits_e, dim=1)

        l_pos_lab_e = torch.einsum('nc,nc->n', [q_extreme_lab, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_lab_e = torch.einsum('nc,ck->nk', [q_extreme_lab, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_lab_e = torch.cat([l_pos_lab_e, l_neg_lab_e], dim=1)
        logits_lab_e /= self.T
        logits_e_ddm = logits_lab_e

        # Compute logits_ed of dropped extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)
        # apply temperature
        logits_ed /= self.T
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()
        
        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, logits_e, logits_ed, labels_ddm, labels_ddm2
    def forward_mutalddm_withcos(self, im_q_extreme, im_q, im_k=None, nnm=False, topk=1,mask=None):
        """
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_extreme: a batch of extremely augmented query sequences
        """
        if mask != None:
            im_mask = im_q.permute(0,2,4,3,1)#NTMVC 
            im_mask = im_mask*mask
            im_mask[mask==0] = self.mask_param
            im_mask = im_mask.permute(0,4,1,3,2)#NCTVM
        

        if nnm:
            return self.nearest_neighbors_mining_mutalddm_withcos(im_q, im_k, im_q_extreme, topk)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        if mask==None:
            q_extreme, q_extreme_drop = self.encoder_q(im_q_extreme, drop_graph=True)  # NxC
        else:
            q_extreme, q_extreme_drop = self.encoder_q(im_mask, drop_graph=True)

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        cos_loss = loss_fn(q_extreme,q)/2
        cos_loss += loss_fn(q_extreme_drop,q_extreme)/2

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        # Compute logits of normally augmented query using Einstein sum
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

        # Compute logits_e of extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        # apply temperature
        logits_e /= self.T
        logits_e_ddm = logits_e
        logits_e = torch.softmax(logits_e, dim=1)

        # Compute logits_ed of dropped extremely augmented query using Einstein sum
        # positive logits: Nx1
        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)
        # apply temperature
        logits_ed /= self.T
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()
        
        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, logits_e, logits_ed, labels_ddm, labels_ddm2, cos_loss
    def nearest_neighbors_mining(self, im_q, im_k, im_q_extreme, topk=1):

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        q_extreme, q_extreme_drop = self.encoder_q(im_q_extreme, drop_graph=True)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])

        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)

        logits /= self.T
        logits_e /= self.T
        logits_ed /= self.T

        logits_e = torch.softmax(logits_e, dim=1)
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()

        # nearest neighbors mining to expand the positive set
        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_e, topk, dim=1)
        _, topkdix_ed = torch.topk(l_neg_ed, topk, dim=1)

        topk_onehot = torch.zeros_like(l_neg)

        topk_onehot.scatter_(1, topkdix, 1)
        topk_onehot.scatter_(1, topkdix_e, 1)
        topk_onehot.scatter_(1, topkdix_ed, 1)

        pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)

        self._dequeue_and_enqueue(k)

        return logits, pos_mask, logits_e, logits_ed, labels_ddm    
    def nearest_neighbors_mining_cossim(self, im_q, im_k, im_q_extreme, topk=1):

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        q_extreme, q_extreme_drop = self.encoder_q(im_q_extreme, drop_graph=True)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        cos_loss = loss_fn(q_extreme,k)/2
        cos_loss += loss_fn(q_extreme_drop,k)/2

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        # Use the distribution of normally augmented view as supervision label

        # nearest neighbors mining to expand the positive set
        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_e, topk, dim=1)
        _, topkdix_ed = torch.topk(l_neg_ed, topk, dim=1)

        topk_onehot = torch.zeros_like(l_neg)

        topk_onehot.scatter_(1, topkdix, 1)
        topk_onehot.scatter_(1, topkdix_e, 1)
        topk_onehot.scatter_(1, topkdix_ed, 1)

        pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)

        self._dequeue_and_enqueue(k)

        return logits, pos_mask, cos_loss
    def nearest_neighbors_mining_cossim_hierarchical(self, im_q, im_k, im_q_extreme, topk=1):

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        q_extreme, q_extreme_drop = self.encoder_q(im_q_extreme, drop_graph=True)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            k2 = self.encoder_k(im_q_extreme)  # keys: NxC
            k2 = F.normalize(k2, dim=1)


        cos_loss = loss_fn(q_extreme,k)/2
        cos_loss += loss_fn(q_extreme_drop,k2)/2

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        # Use the distribution of normally augmented view as supervision label

        # nearest neighbors mining to expand the positive set
        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_e, topk, dim=1)
        _, topkdix_ed = torch.topk(l_neg_ed, topk, dim=1)

        topk_onehot = torch.zeros_like(l_neg)

        topk_onehot.scatter_(1, topkdix, 1)
        topk_onehot.scatter_(1, topkdix_e, 1)
        topk_onehot.scatter_(1, topkdix_ed, 1)

        pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)

        self._dequeue_and_enqueue(k)

        return logits, pos_mask, cos_loss
    def nearest_neighbors_mining_mutalddm(self, im_q, im_k, im_q_extreme, topk=1):

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        q_extreme, q_extreme_drop = self.encoder_q(im_q_extreme, drop_graph=True)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])

        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)

        logits /= self.T
        logits_e /= self.T
        logits_e_ddm = logits_e

        logits_ed /= self.T

        logits_e = torch.softmax(logits_e, dim=1)
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()

        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # nearest neighbors mining to expand the positive set
        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_e, topk, dim=1)
        _, topkdix_ed = torch.topk(l_neg_ed, topk, dim=1)

        topk_onehot = torch.zeros_like(l_neg)

        topk_onehot.scatter_(1, topkdix, 1)
        topk_onehot.scatter_(1, topkdix_e, 1)
        topk_onehot.scatter_(1, topkdix_ed, 1)

        pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)

        self._dequeue_and_enqueue(k)

        return logits, pos_mask, logits_e, logits_ed, labels_ddm, labels_ddm2
    def nearest_neighbors_mining_mutalddm_multinm(self, im_q, im_k, im_q_extreme, topk=1):

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        q_extreme, q_extreme_drop = self.encoder_q(im_q_extreme, drop_graph=True)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])

        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)

        logits /= self.T
        logits_e /= self.T
        logits_e_ddm = logits_e

        logits_ed /= self.T

        logits_e = torch.softmax(logits_e, dim=1)
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()

        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # nearest neighbors mining to expand the positive set
        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_e, topk, dim=1)
        _, topkdix_ed = torch.topk(l_neg_ed, topk, dim=1)

        thresh_idx = (l_neg>0.9)#N,K
        thresh_idx_e = (l_neg_e>0.9)#N,K
        thresh_idx_ed = (l_neg_ed>0.9)#N,K

        topk_onehot = torch.zeros_like(l_neg)#N,K

        topk_onehot.scatter_(1, topkdix, 1)
        topk_onehot.scatter_(1, topkdix_e, 1)
        topk_onehot.scatter_(1, topkdix_ed, 1)
        
        topk_onehot[thresh_idx] = 1
        topk_onehot[thresh_idx_e] = 1
        topk_onehot[thresh_idx_ed] = 1

        pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)

        self._dequeue_and_enqueue(k)

        return logits, pos_mask, logits_e, logits_ed, labels_ddm, labels_ddm2
    def nearest_neighbors_mining_mutalddm_cosnnm(self, im_q, im_k, im_q_extreme, topk=1):

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        q_extreme, q_extreme_drop = self.encoder_q(im_q_extreme, drop_graph=True)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])

        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)

        logits /= self.T
        logits_e /= self.T
        logits_e_ddm = logits_e

        logits_ed /= self.T

        logits_e = torch.softmax(logits_e, dim=1)
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()

        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # nearest neighbors mining to expand the positive set
        lt, topkdix = torch.topk(l_neg, topk, dim=1)#N,1
        let, topkdix_e = torch.topk(l_neg_e, topk, dim=1)
        ledt, topkdix_ed = torch.topk(l_neg_ed, topk, dim=1)


        topk_onehot = torch.zeros_like(l_neg)

        topk_onehot.scatter_(1, topkdix, lt.detach())
        topk_onehot.scatter_(1, topkdix_e, let.detach())
        topk_onehot.scatter_(1, topkdix_ed, ledt.detach())

        pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)

        self._dequeue_and_enqueue(k)

        return logits, pos_mask, logits_e, logits_ed, labels_ddm, labels_ddm2
    def nearest_neighbors_mining_mutalddm_with4p(self, im_q, im_k, im_q_extreme1,im_q_extreme2, topk=1):

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        q_extreme = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC
        q_extreme_drop = self.encoder_q(im_q_extreme2, drop_graph=True,return1d=True)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])

        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)

        logits /= self.T
        logits_e /= self.T
        logits_e_ddm = logits_e

        logits_ed /= self.T

        logits_e = torch.softmax(logits_e, dim=1)
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()

        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # nearest neighbors mining to expand the positive set
        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_e, topk, dim=1)
        _, topkdix_ed = torch.topk(l_neg_ed, topk, dim=1)

        topk_onehot = torch.zeros_like(l_neg)

        topk_onehot.scatter_(1, topkdix, 1)
        topk_onehot.scatter_(1, topkdix_e, 1)
        topk_onehot.scatter_(1, topkdix_ed, 1)

        pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)

        self._dequeue_and_enqueue(k)

        return logits, pos_mask, logits_e, logits_ed, labels_ddm, labels_ddm2
    def nearest_neighbors_mining_mutalddm_with4p_adain(self, im_q, im_k, im_q_extreme1,im_q_extreme2, topk=1,randenque=False):

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        q_extreme = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC
        N = q.shape[0]
        idx = torch.arange(N)
        n = torch.randint(1, N - 1, (1,))
        randidx = (idx + n) % N
        im_q_extreme2 = AdaIN(im_q_extreme2, im_q_extreme2[randidx])
        q_extreme_drop = self.encoder_q(im_q_extreme2, drop_graph=False)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            if randenque:
                idx_k = torch.arange(N)
                n_k = torch.randint(1, N - 1, (1,))
                randidx_k = (idx_k + n_k) % N
                im_q_extreme2_k = AdaIN(im_q_extreme2, im_q_extreme2[randidx_k])
                k_str = self.encoder_k(im_q_extreme2_k) 
                k_str = F.normalize(k_str, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])

        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)

        logits /= self.T
        logits_e /= self.T
        logits_e_ddm = logits_e

        logits_ed /= self.T

        logits_e = torch.softmax(logits_e, dim=1)
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()

        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # nearest neighbors mining to expand the positive set
        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_e, topk, dim=1)
        _, topkdix_ed = torch.topk(l_neg_ed, topk, dim=1)

        topk_onehot = torch.zeros_like(l_neg)

        topk_onehot.scatter_(1, topkdix, 1)
        topk_onehot.scatter_(1, topkdix_e, 1)
        topk_onehot.scatter_(1, topkdix_ed, 1)

        pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)

        if randenque:
            self._dequeue_and_enqueue_wstrong(k, k_str)
        else:
            self._dequeue_and_enqueue(k)

        return logits, pos_mask, logits_e, logits_ed, labels_ddm, labels_ddm2
    def nearest_neighbors_mining_mutalddm_with4p_adain_wrandenque(self, im_q, im_k, im_q_extreme1,im_q_extreme2, im_k_str,topk=1,randenque=False):

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        q_extreme = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC
        N = q.shape[0]
        idx = torch.arange(N)
        n = torch.randint(1, N - 1, (1,))
        randidx = (idx + n) % N
        im_q_extreme2 = AdaIN(im_q_extreme2, im_q_extreme2[randidx])
        q_extreme_drop = self.encoder_q(im_q_extreme2, drop_graph=False)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            if randenque:
                k_str = self.encoder_k(im_k_str)  # keys: NxC
                k_str = F.normalize(k_str, dim=1)
                

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])

        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)

        logits /= self.T
        logits_e /= self.T
        logits_e_ddm = logits_e

        logits_ed /= self.T

        logits_e = torch.softmax(logits_e, dim=1)
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()

        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # nearest neighbors mining to expand the positive set
        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_e, topk, dim=1)
        _, topkdix_ed = torch.topk(l_neg_ed, topk, dim=1)

        topk_onehot = torch.zeros_like(l_neg)

        topk_onehot.scatter_(1, topkdix, 1)
        topk_onehot.scatter_(1, topkdix_e, 1)
        topk_onehot.scatter_(1, topkdix_ed, 1)

        pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)

        if randenque:
            self._dequeue_and_enqueue_wstrong(k, k_str)
        else:
            self._dequeue_and_enqueue(k)

        return logits, pos_mask, logits_e, logits_ed, labels_ddm, labels_ddm2
    def nearest_neighbors_mining_mutalddm_with4p_ablmask(self, im_q, im_k, im_q_extreme1,im_q_extreme2, topk=1,mask=None, im_k_str=None):

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        q_extreme = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC
        q_extreme_drop = self.encoder_q(im_q_extreme2, drop_graph=True,return1d=True)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            if im_k_str!=None:
                k_str = self.encoder_k(im_k_str)  # keys: NxC
                k_str = F.normalize(k_str, dim=1)  

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])

        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)

        logits /= self.T
        logits_e /= self.T
        logits_e_ddm = logits_e

        logits_ed /= self.T

        logits_e = torch.softmax(logits_e, dim=1)
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()

        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # nearest neighbors mining to expand the positive set
        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_e, topk, dim=1)
        _, topkdix_ed = torch.topk(l_neg_ed, topk, dim=1)

        topk_onehot = torch.zeros_like(l_neg)

        topk_onehot.scatter_(1, topkdix, 1)
        topk_onehot.scatter_(1, topkdix_e, 1)
        topk_onehot.scatter_(1, topkdix_ed, 1)

        pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)

        if im_k_str!=None:
            self._dequeue_and_enqueue_wstrong(k,k_str)
        else:
            self._dequeue_and_enqueue(k)

        return logits, pos_mask, logits_e, logits_ed, labels_ddm, labels_ddm2
    def nearest_neighbors_mining_mutalddm_with4p_ablmask_cos(self, im_q, im_k, im_q_extreme1,im_q_extreme2, topk=1,mask=None, im_k_str=None):

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        q_extreme = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC
        q_extreme_drop = self.encoder_q(im_q_extreme2, drop_graph=True,return1d=True)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        cos_loss = loss_fn(q_extreme, q.detach())
        cos_loss += loss_fn(q_extreme_drop, q_extreme.detach())

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            if im_k_str!=None:
                k_str = self.encoder_k(im_k_str)  # keys: NxC
                k_str = F.normalize(k_str, dim=1)  

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        # nearest neighbors mining to expand the positive set
        _, topkdix = torch.topk(l_neg, topk, dim=1)

        topk_onehot = torch.zeros_like(l_neg)
        topk_onehot.scatter_(1, topkdix, 1)

        pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)

        if im_k_str!=None:
            self._dequeue_and_enqueue_wstrong(k,k_str)
        else:
            self._dequeue_and_enqueue(k)

        return logits, pos_mask, cos_loss
    def nearest_neighbors_mining_mutalddm_abl_onlystr(self, im_q, im_k, im_q_extreme1, topk=1):

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        q_extreme = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC
       
        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
       
        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])

       
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
       
        logits /= self.T
        logits_e /= self.T

        logits_e = torch.softmax(logits_e, dim=1)
   
        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()


        # nearest neighbors mining to expand the positive set
        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_e, topk, dim=1)

        topk_onehot = torch.zeros_like(l_neg)

        topk_onehot.scatter_(1, topkdix, 1)
        topk_onehot.scatter_(1, topkdix_e, 1)

        pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)

        self._dequeue_and_enqueue(k)

        return logits, pos_mask, logits_e, labels_ddm
    def nearest_neighbors_mining_mutalddm_with4p_randenque(self, im_q, im_k,im_k_str, im_q_extreme1,im_q_extreme2, topk=1):

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        q_extreme = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC
        q_extreme_drop = self.encoder_q(im_q_extreme2, drop_graph=True,return1d=True)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            k_str = self.encoder_k(im_k_str)  # keys: NxC
            k_str = F.normalize(k_str, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])

        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)

        logits /= self.T
        logits_e /= self.T
        logits_e_ddm = logits_e

        logits_ed /= self.T

        logits_e = torch.softmax(logits_e, dim=1)
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()

        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # nearest neighbors mining to expand the positive set
        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_e, topk, dim=1)
        _, topkdix_ed = torch.topk(l_neg_ed, topk, dim=1)

        topk_onehot = torch.zeros_like(l_neg)

        topk_onehot.scatter_(1, topkdix, 1)
        topk_onehot.scatter_(1, topkdix_e, 1)
        topk_onehot.scatter_(1, topkdix_ed, 1)

        pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)

        self._dequeue_and_enqueue_wstrong(k,k_str)

        return logits, pos_mask, logits_e, logits_ed, labels_ddm, labels_ddm2
    def nearest_neighbors_mining_mutalddm_with4p_doubleque(self, im_q, im_k,im_k_str, im_q_extreme1,im_q_extreme2, topk=1):

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        q_extreme = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC
        q_extreme_drop = self.encoder_q(im_q_extreme2, drop_graph=True,return1d=True)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            k_str = self.encoder_k(im_k_str)  # keys: NxC
            k_str = F.normalize(k_str, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])

        l_pos_e_tar = torch.einsum('nc,nc->n', [q_extreme, k_str]).unsqueeze(-1)
        l_neg_e_tar = torch.einsum('nc,ck->nk', [q_extreme, self.queue_str.clone().detach()])

        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k_str]).unsqueeze(-1)
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue_str.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        logits_e_tar = torch.cat([l_pos_e_tar, l_neg_e_tar], dim=1)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)

        logits /= self.T
        logits_e /= self.T
        logits_e_tar /= self.T
        logits_e_ddm = logits_e_tar

        logits_ed /= self.T

        logits_e = torch.softmax(logits_e, dim=1)
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()

        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # nearest neighbors mining to expand the positive set
        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_e, topk, dim=1)
        _, topkdix_ed = torch.topk(l_neg_ed, topk, dim=1)

        topk_onehot = torch.zeros_like(l_neg)

        topk_onehot.scatter_(1, topkdix, 1)
        topk_onehot.scatter_(1, topkdix_e, 1)
        topk_onehot.scatter_(1, topkdix_ed, 1)

        pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)

        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue_str(k_str)

        return logits, pos_mask, logits_e, logits_ed, labels_ddm, labels_ddm2
    def nearest_neighbors_mining_mutalddm_with5p_randenque(self, im_q, im_k,im_k_str, im_q_extreme1,im_q_extreme2, topk=1):

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        q_extreme = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC
        q_extreme_drop = self.encoder_q(im_q_extreme2, drop_graph=True,return1d=True)  # NxC
        N = q.shape[0]
        idx = torch.arange(N)
        n = torch.randint(1, N - 1, (1,))
        randidx = (idx + n) % N
        im_q_extreme3 = AdaIN(im_k_str, im_k_str[randidx],dim='V')
        q_extreme_drop3 = self.encoder_q(im_q_extreme3, drop_graph=True,return1d=True)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)
        q_extreme_drop3 = F.normalize(q_extreme_drop3, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
            k_str = self.encoder_k(im_k_str)  # keys: NxC
            k_str = F.normalize(k_str, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])

        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])

        l_pos_edd = torch.einsum('nc,nc->n', [q_extreme_drop3, k]).unsqueeze(-1)
        l_neg_edd = torch.einsum('nc,ck->nk', [q_extreme_drop3, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)
        logits_edd = torch.cat([l_pos_edd, l_neg_edd], dim=1)

        logits /= self.T
        logits_e /= self.T
        logits_ed /= self.T
        logits_e_ddm = logits_e
        logits_ed_ddm = logits_ed

        logits_edd /= self.T

        logits_e = torch.softmax(logits_e, dim=1)
        logits_ed = torch.softmax(logits_ed, dim=1)
        logits_edd = torch.softmax(logits_edd, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()

        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        labels_ddm3 = logits_ed_ddm.clone().detach()
        labels_ddm3 = torch.softmax(labels_ddm3, dim=1)
        labels_ddm3 = labels_ddm3.detach()
        # nearest neighbors mining to expand the positive set
        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_e, topk, dim=1)
        _, topkdix_ed = torch.topk(l_neg_ed, topk, dim=1)

        topk_onehot = torch.zeros_like(l_neg)

        topk_onehot.scatter_(1, topkdix, 1)
        topk_onehot.scatter_(1, topkdix_e, 1)
        topk_onehot.scatter_(1, topkdix_ed, 1)

        pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)

        self._dequeue_and_enqueue_wstrong(k,k_str)

        return logits, pos_mask, logits_e, logits_ed, logits_edd, labels_ddm, labels_ddm2, labels_ddm3
    def nearest_neighbors_mining_mutalddm_with4p_mlp2(self, im_q, im_k, im_q_extreme1,im_q_extreme2, topk=1):

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        q = self.last_fc3(q)
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        q_extreme = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC
        q_extreme = self.last_fc2(q_extreme)
        q_extreme_drop = self.encoder_q(im_q_extreme2, drop_graph=True,return1d=True)  # NxC
        q_extreme_drop = self.last_fc1(q_extreme_drop)

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])

        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)

        logits /= self.T
        logits_e /= self.T
        logits_e_ddm = logits_e

        logits_ed /= self.T

        logits_e = torch.softmax(logits_e, dim=1)
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()

        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # nearest neighbors mining to expand the positive set
        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_e, topk, dim=1)
        _, topkdix_ed = torch.topk(l_neg_ed, topk, dim=1)

        topk_onehot = torch.zeros_like(l_neg)

        topk_onehot.scatter_(1, topkdix, 1)
        topk_onehot.scatter_(1, topkdix_e, 1)
        topk_onehot.scatter_(1, topkdix_ed, 1)

        pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)

        self._dequeue_and_enqueue(k)

        return logits, pos_mask, logits_e, logits_ed, labels_ddm, labels_ddm2
    def nearest_neighbors_mining_mutalddm_with4p_mlp(self, im_q, im_k, im_q_extreme1,im_q_extreme2, topk=1):

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        q_extreme_lab = self.encoder_q(im_q_extreme1, drop_graph=False)  # NxC
        q_extreme = self.last_fc2(q_extreme_lab)

        q_extreme_drop = self.encoder_q(im_q_extreme2, drop_graph=True,return1d=True)  # NxC
        q_extreme_drop = self.last_fc1(q_extreme_drop)

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_lab = F.normalize(q_extreme_lab, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])

        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)

        logits /= self.T
        logits_e /= self.T
        #logits_e_ddm = logits_e

        logits_ed /= self.T

        #output before mlp as label
        l_pos_lab_e = torch.einsum('nc,nc->n', [q_extreme_lab, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_lab_e = torch.einsum('nc,ck->nk', [q_extreme_lab, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_lab_e = torch.cat([l_pos_lab_e, l_neg_lab_e], dim=1)
        logits_lab_e /= self.T
        logits_e_ddm = logits_lab_e


        logits_e = torch.softmax(logits_e, dim=1)
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()

        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # nearest neighbors mining to expand the positive set
        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_e, topk, dim=1)
        _, topkdix_ed = torch.topk(l_neg_ed, topk, dim=1)

        topk_onehot = torch.zeros_like(l_neg)

        topk_onehot.scatter_(1, topkdix, 1)
        topk_onehot.scatter_(1, topkdix_e, 1)
        topk_onehot.scatter_(1, topkdix_ed, 1)

        pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)

        self._dequeue_and_enqueue(k)

        return logits, pos_mask, logits_e, logits_ed, labels_ddm, labels_ddm2
    def nearest_neighbors_mining_mutalddm_withcos(self, im_q, im_k, im_q_extreme, topk=1):

        # Obtain the normally augmented query feature
        q = self.encoder_q(im_q)  # NxC
        # Obtain the extremely augmented query feature and dropped extremely augmented query feature
        q_extreme, q_extreme_drop = self.encoder_q(im_q_extreme, drop_graph=True)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_extreme = F.normalize(q_extreme, dim=1)
        q_extreme_drop = F.normalize(q_extreme_drop, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)
        
        cos_loss = loss_fn(q_extreme,q)/2
        cos_loss += loss_fn(q_extreme_drop,q_extreme)/2

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_e = torch.einsum('nc,nc->n', [q_extreme, k]).unsqueeze(-1)
        l_neg_e = torch.einsum('nc,ck->nk', [q_extreme, self.queue.clone().detach()])

        l_pos_ed = torch.einsum('nc,nc->n', [q_extreme_drop, k]).unsqueeze(-1)
        l_neg_ed = torch.einsum('nc,ck->nk', [q_extreme_drop, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_e = torch.cat([l_pos_e, l_neg_e], dim=1)
        logits_ed = torch.cat([l_pos_ed, l_neg_ed], dim=1)

        logits /= self.T
        logits_e /= self.T
        logits_e_ddm = logits_e

        logits_ed /= self.T

        logits_e = torch.softmax(logits_e, dim=1)
        logits_ed = torch.softmax(logits_ed, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()

        labels_ddm2 = logits_e_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # nearest neighbors mining to expand the positive set
        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_e, topk, dim=1)
        _, topkdix_ed = torch.topk(l_neg_ed, topk, dim=1)

        topk_onehot = torch.zeros_like(l_neg)

        topk_onehot.scatter_(1, topkdix, 1)
        topk_onehot.scatter_(1, topkdix_e, 1)
        topk_onehot.scatter_(1, topkdix_ed, 1)

        pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)

        self._dequeue_and_enqueue(k)

        return logits, pos_mask, logits_e, logits_ed, labels_ddm, labels_ddm2,cos_loss