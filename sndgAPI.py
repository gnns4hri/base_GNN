from torch_geometric.data import Data
from torch.utils.data import DataLoader
import dgl
import torch
import numpy as np
import datasetGeneration
from nets import pg_rgcn_gat3
import pickle


def collate(sample):
    graphs, feats, labels = map(list, zip(*sample))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels


class SNGNN():
    def __init__(self, device='cpu'):
        self.device = torch.device(device)# For gpu change it to cuda
        self.device2 = torch.device('cpu')
        self.params = pickle.load(open('SNGNN_PARAMETERS.prms', 'rb'), fix_imports=True)
        print(self.params)

        self.GNNmodel = pg_rgcn_gat3.PRGAT3(self.params['num_feats'],
                                            self.params['n_classes'],
                                            self.params['heads'][0],
                                            self.params['num_rels'],
                                            self.params['num_rels'],
                                            self.params['num_hidden'],  #feats hidden
                                            self.params['num_layers'],
                                            self.params['in_drop'],
                                            self.params['F'],  # F.relu?
                                            self.params['alpha'],
                                            bias=True
                                            )

        # self.GNNmodel = gat2.GAT2(None,
        #                           self.params['num_layers'],
        #                           self.params['num_feats'],
        #                           self.params['num_hidden'],  # feats hidden
        #                           self.params['n_classes'],
        #                           self.params['heads'],
        #                           self.params['F'],  # F.relu?
        #                           self.params['in_drop'],
        #                           self.params['attn_drop'],
        #                           self.params['alpha'],
        #                           residual=False
        #                           )
        self.GNNmodel.load_state_dict(torch.load('SNGNN_MODEL.tch', map_location = device))
        self.GNNmodel.to(self.device)
        self.GNNmodel.eval()


    def predict(self, sn_scenario):
        jsonmodel = sn_scenario.to_json()

        net_type = 'rgcn'
        graph_type = 'relational'
        train_dataset = datasetGeneration.SocNavDataset(jsonmodel, mode='train', alt=graph_type, verbose=False)
        train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=collate)

        for batch, data in enumerate(train_dataloader):
            subgraph, feats, labels = data
            feats = feats.to(self.device)
            edge_index_=torch.stack(subgraph.edges()).to(self.device)
            edge_type_=subgraph.edata['rel_type'].squeeze().to(self.device)
            if net_type == 'rgcn':
                data = Data(x=feats.float().to(self.device), edge_index=edge_index_, edge_type=edge_type_)
                logits = self.GNNmodel(data)[0].detach().to(self.device2).numpy()[0]
            elif net_type == 'gat2':
                self.GNNmodel.set_g(subgraph)
                logits = self.GNNmodel(feats.float())[0].detach().to(self.device2).numpy()[0]

            score = logits * 100
            if score > 100:
                score = 100
            elif score < 0:
                score = 0

        return score


class Human():
    def __init__(self, id, xPos, yPos, angle):
        self.id = id
        self.xPos = xPos
        self.yPos = yPos
        self.angle = angle


class Object():
    def __init__(self, id, xPos, yPos, angle):
        self.id = id
        self.xPos = xPos
        self.yPos = yPos
        self.angle = angle


class SNScenario():
    def __init__(self):
        self.room = None
        self.humans = []
        self.objects = []
        self.interactions = []

    def add_room(self, sn_room):
        self.room = sn_room

    def add_human(self, sn_human):
        self.humans.append(sn_human)

    def add_object(self, sn_object):
        self.objects.append(sn_object)

    def add_interaction(self, sn_interactions):
        self.interactions.append(sn_interactions)

    def to_json(self):
        jsonmodel = {}
        jsonmodel['identifier'] = "000000 A"
        # Adding Robot
        jsonmodel['robot'] = {'id': 0}
        # Adding Room
        jsonmodel['room'] = []
        if self.room:
            for i in range(int(len(self.room.keys()) / 2)):
                jsonmodel['room'].append([self.room['x' + str(i)], self.room['y' + str(i)]])
        # Adding humans and objects
        jsonmodel['humans'] = []
        jsonmodel['objects'] = []
        for _human in self.humans:
            human = {}
            human['id'] = int(_human.id)
            human['xPos'] = float(_human.xPos)
            human['yPos'] = float(_human.yPos)
            human['orientation'] = float(_human.angle)
            jsonmodel['humans'].append(human)
        for object in self.objects:
            Object = {}
            Object['id'] = int(object.id)
            Object['xPos'] = float(object.xPos)
            Object['yPos'] = float(object.yPos)
            Object['orientation'] = float(object.angle)
            jsonmodel['objects'].append(Object)
        # Adding links
        jsonmodel['links'] = []
        for interaction in self.interactions:
            link = []
            link.append(int(interaction[0]))
            link.append(int(interaction[1]))
            link.append('interact')
            jsonmodel['links'].append(link)
        jsonmodel['score'] = 0
        return jsonmodel
