# This script creates the dataset of graphs and labels necesary for training. It takes the raw data stored in json files
# inside data directory.

import os
import sys
import json
import copy
from collections import namedtuple
import math

import torch as th
import dgl
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info
import numpy as np

threshold_human_wall = 1.5
limit = 32000  # Limit of graphs to load
path_saves = 'saves/'  # This variable is necessary due tu a bug in dgl.DGLDataset source code

#  human to wall distance
def dist_h_w(h, wall):
    hxpos = float(h['xPos']) / 100.
    hypos = float(h['yPos']) / 100.
    wxpos = float(wall.xpos) / 100.
    wypos = float(wall.ypos) / 100.
    return math.sqrt((hxpos - wxpos) * (hxpos - wxpos) + (hypos - wypos) * (hypos - wypos))

def get_node_descriptor_header():
    # Node Descriptor Table
    node_descriptor_header = ['R', 'H', 'O', 'L', 'W',
                                   'h_dist', 'h_dist2', 'h_ang_sin', 'h_ang_cos', 'h_orient_sin', 'h_orient_cos',
                                   'o_dist', 'o_dist2', 'o_ang_sin', 'o_ang_cos', 'o_orient_sin', 'o_orient_cos',
                                   'r_m_h', 'r_m_h2', 'r_hs', 'r_hs2',
                                   'w_dist', 'w_dist2', 'w_ang_sin', 'w_ang_cos', 'w_orient_sin', 'w_orient_cos']
    return node_descriptor_header


def get_relations():
    rels = {'p_r', 'o_r', 'l_r', 'l_p', 'l_o', 'p_p', 'p_o', 'w_l', 'w_p'}
    # p = person
    # r = robot
    # l = room (lounge)
    # o = object
    # w = wall
    # n = node (generic)
    for e in list(rels):
        rels.add(e[::-1])
    rels.add('self')
    rels = sorted(list(rels))
    num_rels = len(rels)

    return rels, num_rels


def get_features():
    node_types_one_hot = ['robot', 'human', 'object', 'room', 'wall']
    human_metric_features = ['hum_distance', 'hum_distance2', 'hum_angle_sin', 'hum_angle_cos',
                             'hum_orientation_sin', 'hum_orientation_cos', 'hum_robot_sin',
                             'hum_robot_cos']
    object_metric_features = ['obj_distance', 'obj_distance2', 'obj_angle_sin', 'obj_angle_cos',
                              'obj_orientation_sin', 'obj_orientation_cos']
    room_metric_features = ['room_min_human', 'room_min_human2', 'room_humans', 'room_humans2']
    wall_metric_features = ['wall_distance', 'wall_distance2', 'wall_angle_sin', 'wall_angle_cos',
                            'wall_orientation_sin', 'wall_orientation_cos']
    all_features = node_types_one_hot + human_metric_features + object_metric_features + room_metric_features + \
                   wall_metric_features
    feature_dimensions = len(all_features)

    return all_features, feature_dimensions


#################################################################
# Different initialize alternatives:
#################################################################

# So far there is only one alternative implemented that I think is the most complete

def initializeAlt1(data):
    # Initialize variables
    rels, num_rels = get_relations()
    edge_types = []  # List to store the relation of each edge
    edge_norms = []  # List to store the norm of each edge
    max_used_id = 0  # Initialise id counter (0 for the robot)
    closest_human_distance = -1  # Compute closest human distance

    # Compute data for walls
    Wall = namedtuple('Wall', ['dist', 'orientation', 'angle', 'xpos', 'ypos'])
    walls = []
    for wall_index in range(len(data['room']) - 1):
        p1 = np.array(data['room'][wall_index + 0])
        p2 = np.array(data['room'][wall_index + 1])
        dist = np.linalg.norm(p1 - p2)
        iters = int(dist / 400) + 1
        if iters > 1:
            v = (p2 - p1) / iters
            for i in range(iters):
                pa = p1 + v * i
                pb = p1 + v * (i + 1)
                inc2 = pb - pa
                midsp = (pa + pb) / 2
                walls.append(
                    Wall(np.linalg.norm(midsp) / 100., math.atan2(inc2[0], inc2[1]), math.atan2(midsp[0], midsp[1]),
                         midsp[0], midsp[1]))
        else:
            inc = p2 - p1
            midp = (p2 + p1) / 2
            walls.append(
                Wall(np.linalg.norm(inc / 2) / 100., math.atan2(inc[0], inc[1]), math.atan2(midp[0], midp[1]),
                     midp[0], midp[1]))

    # Compute the number of nodes
    # one for the robot + room walls      + humans               + objects          + room(global node)
    n_nodes = 1 + len(walls) + len(data['humans']) + len(data['objects']) + 1

    # Feature dimensions
    all_features, n_features = get_features()
    features = th.zeros(n_nodes, n_features)

    # Nodes variables
    typeMap = dict()
    position_by_id = {}
    src_nodes = []  # List to store source nodes
    dst_nodes = []  # List to store destiny nodes

    # Labels
    labels = th.zeros([1, 1])  # A 1x1 tensor
    labels[0][0] = th.tensor(float(data['score']) / 100.)

    # robot (id 0)
    robot_id = 0
    typeMap[robot_id] = 'r'  # 'r' for 'robot'
    features[robot_id, all_features.index('robot')] = 1.

    # humans
    for h in data['humans']:
        src_nodes.append(h['id'])
        dst_nodes.append(robot_id)
        edge_types.append(rels.index('p_r'))
        edge_norms.append([1. / len(data['humans'])])

        src_nodes.append(robot_id)
        dst_nodes.append(h['id'])
        edge_types.append(rels.index('r_p'))
        edge_norms.append([1.])

        typeMap[h['id']] = 'p'  # 'p' for 'person'
        max_used_id = max(h['id'], max_used_id)
        xpos = float(h['xPos']) / 100.
        ypos = float(h['yPos']) / 100.

        position_by_id[h['id']] = [xpos, ypos]
        distance = math.sqrt(xpos * xpos + ypos * ypos)
        angle = math.atan2(xpos, ypos)
        orientation = float(h['orientation']) / 180. * math.pi
        while orientation > math.pi: orientation -= 2. * math.pi
        while orientation < -math.pi: orientation += 2. * math.pi
        if orientation > math.pi:
            orientation -= math.pi
        elif orientation < -math.pi:
            orientation += math.pi
        # Compute point of view from humans
        if angle > 0:
            angle_hum = (angle - math.pi) - orientation
        else:
            angle_hum = (math.pi + angle) - orientation

        # print(str(math.degrees(angle)) + ' ' + str(math.degrees(orientation)) + ' ' + str(math.degrees(angle_hum)))
        features[h['id'], all_features.index('human')] = 1.
        features[h['id'], all_features.index('hum_distance')] = distance
        features[h['id'], all_features.index('hum_distance2')] = distance * distance
        features[h['id'], all_features.index('hum_angle_sin')] = math.sin(angle)
        features[h['id'], all_features.index('hum_angle_cos')] = math.cos(angle)
        features[h['id'], all_features.index('hum_orientation_sin')] = math.sin(orientation)
        features[h['id'], all_features.index('hum_orientation_cos')] = math.cos(orientation)
        features[h['id'], all_features.index('hum_robot_sin')] = math.sin(angle_hum)
        features[h['id'], all_features.index('hum_robot_cos')] = math.cos(angle_hum)
        if closest_human_distance < 0 or closest_human_distance > distance:
            closest_human_distance = distance

    # objects
    for o in data['objects']:
        src_nodes.append(o['id'])
        dst_nodes.append(robot_id)
        edge_types.append(rels.index('o_r'))
        edge_norms.append([1. / len(data['objects'])])

        src_nodes.append(robot_id)
        dst_nodes.append(o['id'])
        edge_types.append(rels.index('r_p'))
        edge_norms.append([1.])

        typeMap[o['id']] = 'o'  # 'o' for 'object'
        max_used_id = max(o['id'], max_used_id)
        xpos = float(o['xPos']) / 100.
        ypos = float(o['yPos']) / 100.

        position_by_id[o['id']] = [xpos, ypos]
        distance = math.sqrt(xpos * xpos + ypos * ypos)
        angle = math.atan2(xpos, ypos)
        orientation = float(o['orientation']) / 180. * math.pi
        while orientation > math.pi: orientation -= 2. * math.pi
        while orientation < -math.pi: orientation += 2. * math.pi
        features[o['id'], all_features.index('object')] = 1
        features[o['id'], all_features.index('obj_distance')] = distance
        features[o['id'], all_features.index('obj_distance2')] = distance * distance
        features[o['id'], all_features.index('obj_angle_sin')] = math.sin(angle)
        features[o['id'], all_features.index('obj_angle_cos')] = math.cos(angle)
        features[o['id'], all_features.index('obj_orientation_sin')] = math.sin(orientation)
        features[o['id'], all_features.index('obj_orientation_cos')] = math.cos(orientation)

    # Room (Global node)
    max_used_id += 1
    room_id = max_used_id
    # print('Room will be {}'.format(room_id))
    typeMap[room_id] = 'l'  # 'l' for 'room' (lounge)
    features[room_id, all_features.index('room')] = 1.
    features[room_id, all_features.index('room_min_human')] = closest_human_distance
    features[room_id, all_features.index('room_min_human2')] = closest_human_distance * closest_human_distance
    features[room_id, all_features.index('room_humans')] = len(data['humans'])
    features[room_id, all_features.index('room_humans2')] = len(data['humans']) * len(data['humans'])

    # walls
    wids = dict()
    for wall in walls:
        max_used_id += 1
        wall_id = max_used_id
        wids[wall] = wall_id
        typeMap[wall_id] = 'w'  # 'w' for 'walls'

        src_nodes.append(wall_id)
        dst_nodes.append(room_id)
        edge_types.append(rels.index('w_l'))
        edge_norms.append([1. / len(walls)])

        src_nodes.append(room_id)
        dst_nodes.append(wall_id)
        edge_types.append(rels.index('l_w'))
        edge_norms.append([1.])

        position_by_id[wall_id] = [wall.xpos / 100., wall.ypos / 100.]
        features[wall_id, all_features.index('wall')] = 1.
        features[wall_id, all_features.index('wall_distance')] = wall.dist
        features[wall_id, all_features.index('wall_distance2')] = wall.dist * wall.dist
        features[wall_id, all_features.index('wall_angle_sin')] = math.sin(wall.angle)
        features[wall_id, all_features.index('wall_angle_cos')] = math.cos(wall.angle)
        features[wall_id, all_features.index('wall_orientation_sin')] = math.sin(wall.orientation)
        features[wall_id, all_features.index('wall_orientation_cos')] = math.cos(wall.orientation)

    for h in data['humans']:
        number = 0
        for wall in walls:
            dist = dist_h_w(h, wall)
            if dist < threshold_human_wall:
                number -= - 1
        for wall in walls:
            dist = dist_h_w(h, wall)
            if dist < threshold_human_wall:
                src_nodes.append(wids[wall])
                dst_nodes.append(h['id'])
                edge_types.append(rels.index('w_p'))
                edge_norms.append([1. / number])

    for wall in walls:
        number = 0
        for h in data['humans']:
            dist = dist_h_w(h, wall)
            if dist < threshold_human_wall:
                number -= - 1
        for h in data['humans']:
            dist = dist_h_w(h, wall)
            if dist < threshold_human_wall:
                src_nodes.append(h['id'])
                dst_nodes.append(wids[wall])
                edge_types.append(rels.index('p_w'))
                edge_norms.append([1. / number])

    # interaction links
    for link in data['links']:
        typeLdir = typeMap[link[0]] + '_' + typeMap[link[1]]
        typeLinv = typeMap[link[1]] + '_' + typeMap[link[0]]

        src_nodes.append(link[0])
        dst_nodes.append(link[1])
        edge_types.append(rels.index(typeLdir))
        edge_norms.append([1.])

        src_nodes.append(link[1])
        dst_nodes.append(link[0])
        edge_types.append(rels.index(typeLinv))
        edge_norms.append([1.])

    # Edges for the room node (Global Node)
    for node_id in range(n_nodes):
        typeLdir = typeMap[room_id] + '_' + typeMap[node_id]
        typeLinv = typeMap[node_id] + '_' + typeMap[room_id]
        if node_id == room_id:
            continue

        src_nodes.append(room_id)
        dst_nodes.append(node_id)
        edge_types.append(rels.index(typeLdir))
        edge_norms.append([1.])

        src_nodes.append(node_id)
        dst_nodes.append(room_id)
        edge_types.append(rels.index(typeLinv))
        edge_norms.append([1. / max_used_id])

    # self edges
    for node_id in range(n_nodes - 1):
        src_nodes.append(node_id)
        dst_nodes.append(node_id)
        edge_types.append(rels.index('self'))
        edge_norms.append([1.])

    # Convert outputs to tensors
    edge_types = th.LongTensor(edge_types)
    edge_norms = th.Tensor(edge_norms)

    return src_nodes, dst_nodes, n_nodes, features, edge_types, edge_norms, position_by_id, typeMap, labels


#################################################################
# Class to load the dataset
#################################################################

class SocNavDataset(DGLDataset):
    def __init__(self, path, alt, mode='train', raw_dir='data/', init_line=-1, end_line=-1, loc_limit=limit,
                 force_reload=False, verbose=True, debug=False):
        if type(path) is str:
            self.path = raw_dir + path
        else:
            self.path = path
        self.mode = mode
        self.alt = alt
        self.init_line = init_line
        self.end_line = end_line
        self.graphs = []
        self.labels = []
        self.data = dict()
        self.data['typemaps'] = []
        self.data['coordinates'] = []
        self.data['identifiers'] = []
        self.debug = debug
        self.limit = loc_limit

        # Define device. GPU if it is available
        self.device = 'cpu'

        if self.debug:
            self.limit = 1 + (0 if init_line == -1 else init_line)

        super(SocNavDataset, self).__init__("SocNav", raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)

    def get_dataset_name(self):
        graphs_path = 'graphs_' + self.mode + '_s_' + str(limit) + '.bin'
        info_path = 'info_' + self.mode + '_s_' + str(limit) + '.pkl'
        return graphs_path, info_path

    def generate_final_graph(self, raw_data):
        if self.alt == '1':
            src_nodes, dst_nodes, n_nodes, features, edge_types, edge_norms, position_by_id, typeMap, labels = \
                initializeAlt1(raw_data)
        elif self.alt == '2':
            print('Alternative not yet implemented.')
            sys.exit(0)
        else:
            print('Introduce a valid initialize alternative')
            sys.exit(-1)

        self.data['typemaps'].append(typeMap)
        self.data['coordinates'].append(position_by_id)
        self.data['identifiers'].append(raw_data['identifier'])
        self.data['descriptor_header'] = get_node_descriptor_header()

        self.labels.append(labels)

        try:
            final_graph = dgl.graph((src_nodes, dst_nodes), num_nodes=n_nodes, idtype=th.int32, device=self.device)
            final_graph.ndata['h'] = features
            final_graph.edata.update({'rel_type': edge_types, 'norm': edge_norms})
            return final_graph
        except Exception:
            raise

    #################################################################
    # Implementation of abstract methods
    #################################################################

    def download(self):
        # No need to download any data
        pass

    def process(self):

        if type(self.path) is str and self.path.endswith('.json'):
            linen = -1
            for line in open(self.path).readlines():
                if linen % 1000 == 0:
                    print(linen)

                if linen + 1 >= self.limit:
                    print('Stop including more samples to speed up dataset loading')
                    break
                linen += 1
                if self.init_line >= 0 and linen < self.init_line:
                    continue
                if linen > self.end_line >= 0:
                    continue

                raw_data = json.loads(line)
                final_graph = self.generate_final_graph(raw_data)
                self.graphs.append(final_graph)
            self.labels = th.tensor(self.labels, dtype=th.float64)
        elif type(self.path) == list and type(self.path[0]) == str:
            raw_data = json.loads(self.path)
            final_graph = self.generate_final_graph(raw_data)
            self.graphs.append(final_graph)
            self.labels = th.tensor(self.labels, dtype=th.float64)
        else:
            final_graph = self.generate_final_graph(self.path)
            self.graphs.append(final_graph)
            self.labels = th.tensor(self.labels, dtype=th.float64)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)

    def save(self):
        if self.debug:
            return
        # Generate paths
        graphs_path, info_path = tuple((path_saves + x) for x in self.get_dataset_name())
        os.makedirs(os.path.dirname(path_saves), exist_ok=True)

        # Save graphs
        save_graphs(graphs_path, self.graphs, {'labels': self.labels})

        # Save additional info
        save_info(info_path, {'typemaps': self.data['typemaps'],
                              'coordinates': self.data['coordinates'],
                              'identifiers': self.data['identifiers'],
                              'descriptor_header': self.data['descriptor_header']})

    def load(self):
        # Generate paths
        graphs_path, info_path = tuple((path_saves + x) for x in self.get_dataset_name())

        # Load graphs
        self.graphs, label_dict = load_graphs(graphs_path)
        self.labels = label_dict['labels']

        # Load info
        self.data['typemaps'] = load_info(info_path)['typemaps']
        self.data['coordinates'] = load_info(info_path)['coordinates']
        self.data['descriptor_header'] = load_info(info_path)['descriptor_header']
        self.data['identifiers'] = load_info(info_path)['identifiers']

    def has_cache(self):
        # Generate paths
        graphs_path, info_path = tuple((path_saves + x) for x in self.get_dataset_name())
        if self.debug:
            return False
        return os.path.exists(graphs_path) and os.path.exists(info_path)
