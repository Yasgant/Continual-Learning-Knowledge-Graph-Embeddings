#!/usr/bin/env python
# coding: utf-8

# In[33]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import marius as m
from marius.tools.preprocess.dataset import LinkPredictionDataset
from pathlib import Path
from marius.tools.preprocess.converters.torch_converter import TorchEdgeListConverter
from marius.tools.preprocess.datasets.fb15k_237 import FB15K237
from omegaconf import OmegaConf
import spdlog as log
logger = log.ConsoleLogger('test', False, True, False)

for ewc_lambda in [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]:
    for new_ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # In[34]:


        embed_size = 100
        batch_size = 1024
        total_epoch = 10
        #new_ratio = .1
        lr = .1
        #ewc_lambda = 1e5
        ewc_type = 'ewc' # none, l2, ewc
        os.environ['CUDA_VISIBLE_DEVICES']='3'
        device = torch.device('cuda')
        new_type = 'random' # strategy_edge, strategy_node, random, whole
        num_neigh_layers = 2
        dataset = 'FB15K237' # FB15K237, YAGO-3SP, IMDB-30SP
        logger.info(f'Embed size: {embed_size}, batch size: {batch_size}, total epoch: {total_epoch}, new ratio: {new_ratio}, lr: {lr}, ewc_lambda: {ewc_lambda}, ewc_type: {ewc_type}, new_type: {new_type}, num_neigh_layers: {num_neigh_layers}, dataset: {dataset}')


        # In[35]:


        def read_edges_csv(filename):
            edges = []
            with open(filename) as f:
                for line in f.readlines():
                    line = line.strip().split(',')
                    edges.append((int(line[0]), int(line[1]), int(line[2])))
            return edges

        def read_dif_csv(filename):
            add, rem = [], []
            with open(filename) as f:
                for line in f.readlines():
                    line = line.strip().split(',')
                    if line[0] == 'add':
                        add.append((int(line[1]), int(line[2]), int(line[3])))
                    else:
                        rem.append((int(line[1]), int(line[2]), int(line[3])))
            return add, rem

        if dataset == 'FB15K237':
            edges = read_edges_csv('train_convert.csv')
            test_edges = read_edges_csv('test_convert.csv')
            valid_edges = read_edges_csv('valid_convert.csv')
            import random
            random.shuffle(edges)
            split = [.0, .5, .6, .7, .75, .8, .85, .9, .95, 1.0]
        elif dataset == 'YAGO-3SP':
            Yedges = []
            Yvalidedges = []
            Ytestedges = []
            adds, rems = [], []
            edges, valid_edges, test_edges = [], [], []
            for i in range(3):
                Yedges.append(read_edges_csv(f'YAGO-3SP/train{i}.csv'))
                Yvalidedges.append(read_edges_csv(f'YAGO-3SP/valid{i}.csv'))
                Ytestedges.append(read_edges_csv(f'YAGO-3SP/test{i}.csv'))
                if i > 0:
                    add, rem = read_dif_csv(f'YAGO-3SP/dif{i}.csv')
                    adds.append(add)
                    rems.append(rem)
        elif dataset == 'IMDB-30SP':
            Iedges = []
            Ivalidedges = []
            Itestedges = []
            adds, rems = [], []
            edges, valid_edges, test_edges = [], [], []
            for i in range(30):
                Iedges.append(read_edges_csv(f'IMDB-30SP/train{i}.csv'))
                Ivalidedges.append(read_edges_csv(f'IMDB-30SP/valid{i}.csv'))
                Itestedges.append(read_edges_csv(f'IMDB-30SP/test{i}.csv'))
                if i > 0:
                    add, rem = read_dif_csv(f'IMDB-30SP/dif{i}.csv')
                    adds.append(add)
                    rems.append(rem)
        else:
            raise Exception('dataset not supported')


        # In[36]:


        if dataset == 'FB15K237':
            nrelation = 237
            nentity = 14541
        elif dataset == 'YAGO-3SP':
            nrelation = 37
            nentity = 27136
        elif dataset == 'IMDB-30SP':
            nrelation = 14
            nentity = 312588
        logger.info(f'{nentity}, {nrelation}')


        # In[37]:


        def init_model(embedding_dim, num_nodes, num_relations, device, dtype):
            # setup shallow embedding encoder
            embedding_layer = m.nn.layers.EmbeddingLayer(dimension=embedding_dim, device=device)
            encoder = m.encoders.GeneralEncoder(layers=[[embedding_layer]])

            # initialize node embedding table
            emb_table = embedding_layer.init_embeddings(num_nodes)

            # initialize DistMult decoder
            decoder = m.nn.decoders.edge.DistMult(num_relations=num_relations,
                                                embedding_dim=embedding_dim,
                                                use_inverse_relations=True,
                                                device=device,
                                                dtype=dtype,
                                                mode="train")

            loss = m.nn.SoftmaxCrossEntropy(reduction="sum")

            # metrics to compute during evaluation
            reporter = m.report.LinkPredictionReporter()
            reporter.add_metric(m.report.MeanReciprocalRank())
            reporter.add_metric(m.report.MeanRank())
            reporter.add_metric(m.report.Hitsk(1))
            reporter.add_metric(m.report.Hitsk(10))

            # sparse_lr sets the learning rate for the embedding parameters
            model = m.nn.Model(encoder, decoder, loss, reporter, sparse_lr=lr)

            # set optimizer for dense model parameters. In this case this is the DistMult relation (edge-type) embeddings
            model.optimizers = [m.nn.AdamOptimizer(model.named_parameters(), lr=lr)]

            return model, emb_table


        # In[38]:


        def get_loss(model, batch):
            scores = model.forward_lp(batch, True)
            loss1 = model.loss_function(scores[0], scores[1], True)
            loss2 = model.loss_function(scores[2], scores[3], True)
            return loss1 + loss2

        def train_epoch(model, embeddings, train_edges, first_train, ewc):
            #print('start epoch...')
            train_neg_sampler = m.data.samplers.CorruptNodeNegativeSampler(num_chunks=10, num_negatives=500, degree_fraction=0.0, filtered=True)
            dataloader = m.data.DataLoader(edges=torch.tensor(train_edges,dtype=torch.int32, device=device),
                                                node_embeddings=embeddings,
                                                batch_size=batch_size,
                                                neg_sampler=train_neg_sampler,
                                                filter_edges=[torch.tensor(edges, dtype=torch.int32, device=device)],
                                                learning_task="lp",
                                                train=True)
            dataloader.initializeBatches()

            counter = 0
            loss_function = model.loss_function
            while dataloader.hasNextBatch():

                batch = dataloader.getBatch()
                model.optimizers[0].clear_grad()
                batch.node_embeddings.requires_grad_()
                loss = get_loss(model, batch)
                if not first_train and ewc_type != 'none':
                    loss2 = ewc.get_loss(model.parameters(), batch)
                    #print(loss, loss2)
                    loss = loss + loss2
                loss.backward()
                model.optimizers[0].step()
                batch.accumulateGradients(lr)
                dataloader.updateEmbeddings(batch)

                counter += 1


        # In[39]:


        def eval_epoch(model, embeddings):
            eval_neg_sampler = m.data.samplers.CorruptNodeNegativeSampler(filtered=True)
            dataloader = m.data.DataLoader(edges=torch.tensor(test_edges, dtype=torch.int32, device=device),
                                                node_embeddings=embeddings,
                                                batch_size=batch_size,
                                                neg_sampler=eval_neg_sampler,
                                                learning_task="lp",
                                                filter_edges=[torch.tensor(edges, dtype=torch.int32, device=device), torch.tensor(valid_edges, dtype=torch.int32, device=device), torch.tensor(test_edges, dtype=torch.int32, device=device)], # used to filter out false negatives in evaluation
                                                train=False)
            # need to reset dataloader before state each epoch
            dataloader.initializeBatches()

            counter = 0
            while dataloader.hasNextBatch():

                batch = dataloader.getBatch()
                model.evaluate_batch(batch)

                counter += 1


            model.reporter.report()


        # In[40]:


        from torch import autograd
        class EWC:
            def __init__(self, ewc_lambda):
                self.ewc_lambda = ewc_lambda
            
            def update_mean(self, paras, embeddings):
                self.paras = []
                for para in paras:
                    self.paras.append(para.data.clone())
                self.embeddings = embeddings.data.clone()
            
            def update_fisher(self, paras, embeddings, loss):
                self.paras_fishers = []
                grads = autograd.grad(loss, [embeddings])
                #for i in range(len(paras)):
                #    self.paras_fishers.append(grads[i].data.clone() ** 2)
                self.embeddings_fishers = grads[-1].data.clone() ** 2
            
            def update(self, paras, batch, loss, embeddings):
                self.update_mean(paras, batch.node_embeddings)
                self.update_fisher(paras, batch.node_embeddings, loss)
                self.embeddings = embeddings.clone()
                temp_embeddings = self.embeddings_fishers
                self.embeddings_fishers = torch.zeros(embeddings.shape, device=device, requires_grad=False)
                self.embeddings_fishers[batch.unique_node_indices] = temp_embeddings

            def get_loss(self, paras, batch):
                losses = []
                nodes = batch.unique_node_indices
                #for i in range(len(paras)):
                #    para, saved_para, saved_fisher = paras[i], self.paras[i], self.paras_fishers[i]
                #    if ewc_type == 'ewc':
                #        losses.append((saved_fisher * (para - saved_para) ** 2).sum()/len(para))
                #    else:
                #        losses.append((ewc_lambda * (para - saved_para) ** 2).sum()/len(para))
                if ewc_type == 'ewc':
                    losses.append((self.embeddings_fishers[nodes] * (batch.node_embeddings - self.embeddings[nodes]) ** 2).sum()/len(nodes))
                else:
                    losses.append((ewc_lambda * (batch.node_embeddings - self.embeddings[nodes]) ** 2).sum()/len(nodes))
                return self.ewc_lambda / 2 * sum(losses)


        # In[41]:


        def get_batch(model, embeddings, edges):
            eval_neg_sampler = m.data.samplers.CorruptNodeNegativeSampler(filtered=False)
            dataloader = m.data.DataLoader(edges=torch.tensor(edges, dtype=torch.int32, device=device),
                                                node_embeddings=embeddings,
                                                batch_size=400000,
                                                neg_sampler=eval_neg_sampler,
                                                learning_task="lp",
                                                train=False)
            dataloader.initializeBatches()
            return dataloader.getBatch()

        def forward(model, embeddings, edges):
            batch = get_batch(model, embeddings, edges)
            assert batch is not None
            values = model.forward_lp(batch, train=False)[0].cpu().detach().numpy()
            return batch.edges.cpu().detach().numpy(), values


        # In[42]:


        def get_upper_part_edges(model, embeddings, edges, new_ratio):
            if len(edges)==0:
                return []
            edges, values = forward(model, embeddings, edges)
            values = values[:len(edges)]
            new_size = int(len(edges)*new_ratio)
            threshold = values[np.argpartition(values, - new_size)[-new_size]]
            return list(edges[values > threshold])
        def get_lower_part_edges(model, embeddings, edges, new_ratio):
            if len(edges)==0:
                return []
            edges, values = forward(model, embeddings, edges)
            values = values[:len(edges)]
            new_size = int(len(edges)*new_ratio)
            threshold = values[np.argpartition(values, - new_size)[new_size]]
            return list(edges[values < threshold])
        def get_part_edges(model, embeddings, edges, new_ratio):
            return get_upper_part_edges(model, embeddings, edges, new_ratio/2) +            get_lower_part_edges(model, embeddings, edges, new_ratio/2)


        # In[43]:


        def get_affected_nodes(edges):
            nodes = set()
            for edge in edges:
                nodes |= set([edge[0], edge[2]])
            return nodes
            
        def get_filtered_edges(edges, new_edges):
            nodes = get_affected_nodes(new_edges)
            ans = []
            for edge in edges:
                if edge[0] in nodes or edge[2] in nodes:
                    ans.append(edge)
            return ans

        def get_dis_to_nodes(model, embeddings, new_edges):
            dis = [0 for i in range(nentity+1)]
            new_edges, values = forward(model, embeddings, new_edges)
            for edge, value in zip(new_edges, values):
                dis[edge[0]] += 1/value
                dis[edge[2]] += 1/value
            return dis

        def get_edges_from_dis(edges, dis):
            new_size = int(len(dis)*new_ratio)
            threshold = dis[np.argpartition(dis, - new_size)[-new_size]]
            ans = []
            for edge in edges:
                if dis[edge[0]] > threshold or dis[edge[2]] > threshold:
                    ans.append(edge)
            return ans

        def get_random_nodes():
            import random
            nodes = list(range(nentity+1))
            random.shuffle(nodes)
            return set(nodes[:int(len(nodes)*new_ratio)])

        def get_edges_from_nodes(edges, nodes):
            ans = []
            for edge in edges:
                if edge[0] in nodes or edge[2] in nodes:
                    ans.append(edge)
            return ans


        # In[44]:


        def get_neighbors(nodes):
            for edge in edges:
                if edge[0] in nodes or edge[2] in nodes:
                    nodes |= set([edge[0], edge[2]])
            return nodes

        def get_affected_degree_of_nodes(model, embeddings, edges):
            ans = [0 for i in range(nentity+1)]
            new_edges, values = forward(model, embeddings, edges)
            values = values - min(values) + 1
            for edge, value in zip(new_edges, values):
                ans[edge[0]] += 1/value
                ans[edge[2]] += 1/value
            return ans

        def get_weighted_edges(model, embeddings, new_edges, new_size, old_edges):
            nodes = get_affected_nodes(new_edges)
            for _ in range(num_neigh_layers):
                nodes = get_neighbors(nodes)
            degree = get_affected_degree_of_nodes(model, embeddings, new_edges)
            total_degree = .0
            for edge in old_edges:
                if edge[0] in nodes or edge[2] in nodes:
                    total_degree += degree[edge[0]] + degree[edge[2]]
            ans = []
            for edge in old_edges:
                if np.random.random() < new_size * (degree[edge[0]] + degree[edge[2]])/total_degree:
                    ans.append(edge)
            return ans


        # In[45]:


        model, embeddings = init_model(embed_size, nentity, nrelation, device, torch.float32)


        # In[46]:


        ewc = EWC(ewc_lambda)
        if dataset == 'FB15K237':
            for t in range(1, len(split)):
                old_edges = edges[:int(len(edges)*split[t-1])]
                new_edges = edges[int(len(edges)*split[t-1]):int(len(edges)*split[t])]

                if t > 1:
                    batch = get_batch(model, embeddings, old_edges)
                    batch.node_embeddings.requires_grad_()
                    loss = get_loss(model, batch)
                    if t > 2:
                        assert (old_embeddings == ewc.embeddings).all()
                    ewc.update(model.parameters(), batch, loss, embeddings)
                    delta = 0
                    delta += ((embeddings - old_embeddings) ** 2).sum()
                    delta += ((old_paras[0] - model.parameters()[0]) ** 2).sum()
                    delta += ((old_paras[1] - model.parameters()[1]) ** 2).sum()
                    logger.info(f"delta: {delta}")

                old_embeddings = embeddings.clone()
                old_paras = [x.clone() for x in model.parameters()]
                if new_type == 'strategy_edge':
                    old_edges = get_lower_part_edges(model, embeddings, old_edges, new_ratio)
                    train_edges = old_edges + new_edges
                elif new_type == 'strategy_node':
                    old_edges = get_filtered_edges(old_edges, new_edges)
                    #dis = get_dis_to_nodes(model, embeddings, new_edges)
                    #old_edges = get_edges_from_dis(old_edges, dis)
                    train_edges = old_edges + new_edges
                elif new_type == 'random':
                    import random
                    random.shuffle(old_edges)
                    train_edges = old_edges[:int(len(old_edges)*new_ratio)]+new_edges
                elif new_type == 'whole':
                    train_edges = old_edges + new_edges
                else:
                    nodes = get_random_nodes()
                    old_edges = get_edges_from_nodes(old_edges, nodes)
                    train_edges = old_edges + new_edges
                logger.info(f't:{t}, training size: {len(train_edges)}, total size: {int(len(edges)*split[t])}')

                for epoch in range(total_epoch):
                    train_epoch(model, embeddings, train_edges, t==1, ewc)
                    
                    if epoch == total_epoch-1:
                        eval_epoch(model, embeddings)


        # In[ ]:


        if dataset == 'YAGO-3SP':
            for t in range(3):
                new_edges = []
                if t == 0:
                    new_edges = Yedges[0]
                    old_edges = []
                    valid_edges = Yvalidedges[0]
                    test_edges = Ytestedges[0]
                    edges = new_edges
                else:
                    new_edges = []
                    old_edges = []
                    old_edgeset = set(Yedges[t-1])
                    for edge in Yedges[t]:
                        if edge in old_edgeset:
                            old_edges.append(edge)
                        else:
                            new_edges.append(edge)
                    valid_edges = Yvalidedges[t]
                    test_edges = Ytestedges[t]
                    edges = new_edges + old_edges

                if t >= 1:
                    batch = get_batch(model, embeddings, old_edges)
                    batch.node_embeddings.requires_grad_()
                    loss = get_loss(model, batch)
                    if t >= 2:
                        assert (old_embeddings == ewc.embeddings).all()
                    ewc.update(model.parameters(), batch, loss, embeddings)
                    for para in model.parameters():
                        para.requires_grad_(False)
                    delta = 0
                    delta += ((embeddings - old_embeddings) ** 2).sum()
                    delta += ((old_paras[0] - model.parameters()[0]) ** 2).sum()
                    delta += ((old_paras[1] - model.parameters()[1]) ** 2).sum()
                    logger.info(f"delta: {delta}")

                old_embeddings = embeddings.clone()
                old_paras = [x.clone() for x in model.parameters()]
                if new_type == 'strategy_edge':
                    old_edges = get_lower_part_edges(model, embeddings, old_edges, new_ratio)
                    train_edges = old_edges + new_edges
                elif new_type == 'strategy_node':
                    new_size = int(len(old_edges)*new_ratio)
                    old_edges = get_weighted_edges(model, embeddings, new_edges, new_size, old_edges)
                    train_edges = old_edges + new_edges
                elif new_type == 'random':
                    import random
                    random.shuffle(old_edges)
                    train_edges = old_edges[:int(len(old_edges)*new_ratio)]+new_edges
                elif new_type == 'whole':
                    train_edges = old_edges + new_edges
                else:
                    nodes = get_random_nodes()
                    old_edges = get_edges_from_nodes(old_edges, nodes)
                    train_edges = old_edges + new_edges
                logger.info(f't:{t}, training size: {len(train_edges)}, total size: {int(len(edges))}')

                for epoch in range(total_epoch):
                    train_epoch(model, embeddings, train_edges, t==0, ewc)
                    
                    if epoch == total_epoch-1:
                        eval_epoch(model, embeddings)


        # In[ ]:


        if dataset == 'IMDB-30SP':
            for t in range(30):
                new_edges = []
                if t == 0:
                    new_edges = Iedges[0]
                    old_edges = []
                    valid_edges = Ivalidedges[0]
                    test_edges = Itestedges[0]
                    edges = new_edges
                else:
                    new_edges = []
                    old_edges = []
                    old_edgeset = set(Iedges[t-1])
                    for edge in Iedges[t]:
                        if edge in old_edgeset:
                            old_edges.append(edge)
                        else:
                            new_edges.append(edge)
                    valid_edges = Ivalidedges[t]
                    test_edges = Itestedges[t]
                    edges = new_edges + old_edges

                if t >= 1:
                    batch = get_batch(model, embeddings, old_edges)
                    batch.node_embeddings.requires_grad_()
                    loss = get_loss(model, batch)
                    if t >= 2:
                        assert (old_embeddings == ewc.embeddings).all()
                    ewc.update(model.parameters(), batch, loss, embeddings)
                    for para in model.parameters():
                        para.requires_grad_(False)
                    delta = 0
                    delta += ((embeddings - old_embeddings) ** 2).sum()
                    delta += ((old_paras[0] - model.parameters()[0]) ** 2).sum()
                    delta += ((old_paras[1] - model.parameters()[1]) ** 2).sum()
                    logger.info(f"delta: {delta}")

                old_embeddings = embeddings.clone()
                old_paras = [x.clone() for x in model.parameters()]
                if new_type == 'strategy_edge':
                    old_edges = get_lower_part_edges(model, embeddings, old_edges, new_ratio)
                    train_edges = old_edges + new_edges
                elif new_type == 'strategy_node':
                    new_size = int(len(old_edges)*new_ratio)
                    old_edges = get_weighted_edges(model, embeddings, new_edges, new_size, old_edges)
                    train_edges = old_edges + new_edges
                elif new_type == 'random':
                    import random
                    random.shuffle(old_edges)
                    train_edges = old_edges[:int(len(old_edges)*new_ratio)]+new_edges
                elif new_type == 'whole':
                    train_edges = old_edges + new_edges
                else:
                    nodes = get_random_nodes()
                    old_edges = get_edges_from_nodes(old_edges, nodes)
                    train_edges = old_edges + new_edges
                logger.info(f't:{t}, training size: {len(train_edges)}, total size: {int(len(edges))}')

                for epoch in range(total_epoch):
                    train_epoch(model, embeddings, train_edges, t==0, ewc)
                    
                    if epoch == total_epoch-1:
                        eval_epoch(model, embeddings)

