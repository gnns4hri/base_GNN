import random
import numpy as np
import torch
import dgl
import pickle
import torch.nn.functional as F
from nets.gcn import GCN
from nets.gat import GAT
from nets.gat2 import GAT2
from nets.rgcn import RGCN
from nets.pg_gcn import PGCN
from nets.pg_ggn import GGN
from nets.pg_rgcn import PRGCN
from nets.pg_gat import PGAT
from nets.pg_rgcn_gat import PRGAT
from nets.pg_rgcn_gat2 import PRGAT2
from nets.pg_rgcn_gat3 import PRGAT3
import datasetGeneration
from torch.utils.data import DataLoader
# from sklearn.metrics import f1_score
# from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from torch_geometric.data import Data

if torch.cuda.is_available() is True:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def describe_model(model):
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


def collate(batch):
    graphs, labels = map(list, zip(*batch))
    batched_graphs = dgl.batch(graphs).to(torch.device(device))
    labels = torch.tensor(labels).to(torch.device(device))

    return batched_graphs, labels


def evaluate(feats, model, subgraph, labels, loss_fcn, fw, net_class):
    with torch.no_grad():
        model.eval()
        if fw == 'dgl':
            model.g = subgraph
            for layer in model.layers:
                layer.g = subgraph
            output = model(feats.float())
        else:
            if net_class in [PRGCN, PRGAT, PRGAT2, PRGAT3]:
                data = Data(x=feats.float(), edge_index=torch.stack(subgraph.edges()).to(feats.device),
                            edge_type=subgraph.edata['rel_type'].squeeze().to(feats.device))
            else:
                data = Data(x=feats.float(), edge_index=torch.stack(subgraph.edges()).to(feats.device))
            output = model(data)
        loss_data = loss_fcn(output[getMaskForBatch(subgraph)], labels.float())
        # predict = output.data.cpu().numpy()
        predict = output[getMaskForBatch(subgraph)].data.cpu().numpy()
        #score = mean_absolute_error(labels.data.cpu().numpy(), predict)
        #print('MAE {}'.format(score))
        score = mean_squared_error(labels.data.cpu().numpy(), predict)
        # print('MSE {}'.format(mean_squared_error(labels.data.cpu().numpy(), predict)))
        # score = r2_score(labels.data.cpu().numpy(), predict)
        # print('R2S {}'.format(score))
        # score = f1_score(labels.data.cpu().numpy(), predict, average='micro')
        ##return loss_data.item(), loss_data.item()
        return score, loss_data.item()


def getMaskForBatch(subgraph):
    future_index = 0
    indexes = []
    for g in dgl.unbatch(subgraph):
        indexes.append(future_index)
        future_index += g.number_of_nodes()
    return indexes

def flattenList(input_data):
    def flattenList_rec(input_data):
        if type(input_data) is not list:
            return input_data
        ret = []
        for x in input_data:
            if type(x) is list:
                for y in x:
                    if type(y) is list:
                        ret += flattenList(y)
                    else:
                        ret.append(y)
            else:
                ret.append(x)
        return ret

    ret = flattenList_rec(input_data)
    if type(ret) is not list:
        return [ret]
    return ret

# MAIN


def main(training_file, dev_file, test_file, epochs=None, patience=None, num_heads=None, num_out_heads=None,
         num_layers=None, num_hidden=None, residual=None, in_drop=None, attn_drop=None, lr=None, weight_decay=None,
         alpha=None, batch_size=None, graph_type=None, net=None, cuda=None, fw='dgl'):

    print('FW', fw)
    # number of training epochs
    if epochs is None:
        epochs = 400
    print('EPOCHS', epochs)
    # used for early stop
    if patience is None:
        patience = 15
    print('PATIENCE', patience)

    # number of hidden attention heads
    if num_heads is None:
        num_heads_ch = [4, 5, 6, 7]
    else:
        num_heads_ch = flattenList(num_heads)
    print('NUM HEADS', num_heads_ch)

    # number of output attention heads
    if num_out_heads is None:
        num_out_heads_ch = [ 4, 5, 6, 7]
    else:
        num_out_heads_ch = flattenList(num_out_heads)
    print('NUM OUT HEADS', num_out_heads_ch)

    # number of hidden layers
    if num_layers is None:
        num_layers_ch = [2, 3, 4, 5, 6]
    else:
        num_layers_ch = flattenList(num_layers)
    print('NUM LAYERS', num_layers_ch)
    # number of hidden units
    if num_hidden is None:
        num_hidden_ch = [32, 64, 96, 128, 256, 350, 512]
    else:
        num_hidden_ch = flattenList(num_hidden)
    print('NUM HIDDEN', num_hidden_ch)
    # use residual connection
    if residual is None:
        residual_ch = [True, False]
    else:
        residual_ch = flattenList(residual)
    print('RESIDUAL', residual_ch)
    # input feature dropout
    if in_drop is None:
        in_drop_ch = [0., 0.001, 0.0001, 0.00001]
    else:
        in_drop_ch = flattenList(in_drop)
    print('IN DROP', in_drop_ch)
    # attention dropout
    if attn_drop is None:
        attn_drop_ch = [0., 0.001, 0.0001, 0.00001]
    else:
        attn_drop_ch = flattenList(attn_drop)
    print('ATTENTION DROP', attn_drop_ch)
    # learning rate
    if lr is None:
        lr_ch = [0.0000005,  0.0000015,  0.00001, 0.00005, 0.0001]
    else:
        lr_ch = flattenList(lr)
    print('LEARNING RATE', lr_ch)
    # weight decay
    if weight_decay is None:
        weight_decay_ch = [0.0001, 0.001, 0.005]
    else:
        weight_decay_ch = flattenList(weight_decay)
    print('WEIGHT DECAY', weight_decay_ch)
    # the negative slop of leaky relu
    if alpha is None:
        alpha_ch = [0.1, 0.15, 0.2]
    else:
        alpha_ch = flattenList(alpha)
    print('ALPHA', alpha_ch)
    # batch size used for training, validation and test
    if batch_size is None:
        batch_size_ch = [175, 256, 350, 450, 512, 800, 1600]
    else:
        batch_size_ch = flattenList(batch_size)
    print('BATCH SIZE', batch_size_ch)
    # graph type
    if graph_type is None:
        graph_type_ch = ['1']
    else:
        graph_type_ch = flattenList(graph_type)
    print('GRAPH TYPE', graph_type_ch)
    # net type
    print('NET A ', net)
    if net is None:
        net_ch = [GCN, GAT, GAT2, RGCN, PGCN, PRGCN, GGN, PGAT]
    else:
        net_ch_raw = flattenList(net)
        net_ch = []
        for ch in net_ch_raw:
            if ch.lower() == 'gcn':
                if fw == 'dgl':
                    net_ch.append(GCN)
                else:
                    net_ch.append(PGCN)
            elif ch.lower() == 'gat':
                if fw == 'dgl':
                    net_ch.append(GAT)
                else:
                    net_ch.append(PGAT)
            elif ch.lower() == 'gat2':
                net_ch.append(GAT2)
            elif ch.lower() == 'rgcn':
                if fw == 'dgl':
                    net_ch.append(RGCN)
                else:
                    net_ch.append(PRGCN)
            elif ch.lower() == 'ggn':
                net_ch.append(GGN)
            elif ch.lower() == 'rgat':
                net_ch.append(PRGAT)
            elif ch.lower() == 'rgat2':
                net_ch.append(PRGAT2)
            elif ch.lower() == 'rgat3':
                net_ch.append(PRGAT3)
            else:
                print('Network type {} is not recognised.'.format(ch))
                import sys
                sys.exit(1)
    print('NET TYPE', net_ch)
    # CUDA?
    if cuda is None:
        device = torch.device("cpu")
    elif cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('DEVICE', device)

    # define loss function
    # loss_fcn = torch.nn.BCEWithLogitsLoss()
    loss_fcn = torch.nn.MSELoss()

    num_heads     = random.choice(num_heads_ch)
    num_out_heads = random.choice(num_out_heads_ch)
    num_layers    = random.choice(num_layers_ch)
    num_hidden    = random.choice(num_hidden_ch)
    residual      = random.choice(residual_ch)
    in_drop       = random.choice(in_drop_ch)
    attn_drop     = random.choice(attn_drop_ch)
    lr            = random.choice(lr_ch)
    weight_decay  = random.choice(weight_decay_ch)
    alpha         = random.choice(alpha_ch)
    batch_size    = random.choice(batch_size_ch)
    graph_type    = random.choice(graph_type_ch)
    net_class     = random.choice(net_ch)

    print('=========================')
    print('HEADS',        num_heads)
    print('OUT_HEADS',    num_out_heads)
    print('LAYERS',       num_layers)
    print('HIDDEN',       num_hidden)
    print('RESIDUAL',     residual)
    print('inDROP',       in_drop)
    print('atDROP',       attn_drop)
    print('LR',           lr)
    print('DECAY',        weight_decay)
    print('ALPHA',        alpha)
    print('BATCH',        batch_size)
    print('GRAPH_ALT',    graph_type)
    print('ARCHITECTURE', net_class)
    print('=========================')

    # create the dataset
    print('Loading training set...')
    train_dataset = datasetGeneration.SocNavDataset(training_file, mode='train', alt=graph_type)
    print('Loading dev set...')
    valid_dataset = datasetGeneration.SocNavDataset(dev_file, mode='valid', alt=graph_type)
    print('Loading test set...')
    test_dataset  = datasetGeneration.SocNavDataset(test_file, mode='test', alt=graph_type)
    print('Done loading files')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate)

    _, num_rels = datasetGeneration.get_relations()
    cur_step = 0
    best_loss = -1
    n_classes = 1
    print('Number of classes:  {}'.format(n_classes))
    num_feats = train_dataset.graphs[0].ndata['h'].shape[1]
    print('Number of features: {}'.format(num_feats))
    g = dgl.batch(train_dataset.graphs)
    heads = ([num_heads] * num_layers) + [num_out_heads]
    # define the model

    print('LAST', fw, net_class)

    if fw == 'dgl':
        if net_class in [GCN]:
            model = GCN(g,
                        num_feats,
                        num_hidden,
                        n_classes,
                        num_layers,
                        F.elu,
                        in_drop)
        elif net_class in [GAT, GAT2]:
            print(net_class)
            model = net_class(g,
                              num_layers,
                              num_feats,
                              num_hidden,
                              n_classes,
                              heads,
                              F.elu,
                              in_drop,
                              attn_drop,
                              alpha,
                              residual,
                              )
        elif net_class in [RGCN]:
            model = RGCN(g,
                         in_dim=num_feats,
                         h_dim=num_hidden,
                         out_dim=n_classes,
                         num_rels=num_rels,
                         feat_drop=in_drop,
                         num_hidden_layers=num_layers)
        else:
            print('Unhandled', net_class)
            import sys
            sys.exit(1)
    else:
            if net_class in [PGCN]:
                model = PGCN(num_feats,
                            n_classes,
                            num_hidden,
                            num_layers,
                            in_drop,
                            F.relu,
                            improved=True,#Compute A-hat as A + 2I
                            bias=True)
            elif net_class in [PRGCN]:
                model = PRGCN(num_feats,
                            n_classes,
                            num_rels,
                            num_rels,  # num_rels?   # TODO: Add variable
                            num_hidden,
                            num_layers,
                            in_drop,
                            F.relu,
                            bias=True
                            )
            elif net_class in [PGAT]:
                model = PGAT(num_feats,
                            n_classes,
                            num_heads,
                            in_drop,
                            num_hidden,
                            num_layers,
                            F.relu,
                            concat=True,
                            neg_slope=alpha,
                            bias=True)
            elif net_class in [PRGAT]:
                model = PRGAT(num_feats,
                              n_classes,
                              num_heads,
                              num_rels,
                              num_rels,  # num_rels?   # TODO: Add variable
                              num_hidden,
                              num_layers,
                              num_layers,
                              in_drop,
                              F.relu,
                              alpha,
                              bias=True
                              )
            elif net_class in [PRGAT2]:
                model = PRGAT2(num_feats,
                               n_classes,
                               num_heads,
                               num_rels,
                               num_rels,  # num_rels?   # TODO: Add variable
                               num_hidden,
                               num_layers,
                               in_drop,
                               F.relu,
                               alpha,
                               bias=True
                               )
            elif net_class in [PRGAT3]:
                model = PRGAT3(num_feats,
                               n_classes,
                               num_heads,
                               num_rels,
                               num_rels,  # num_rels?   # TODO: Add variable
                               num_hidden,
                               num_layers,
                               in_drop,
                               F.relu,
                               alpha,
                               bias=True
                               )
            else:
                model = GGN(num_feats,
                            num_layers,
                            aggr='mean',
                            bias=True)
    #Describe the model
    #describe_model(model)

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # for name, param in model.named_parameters():
        # if param.requires_grad:
        # print(name, param.data.shape)
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        loss_list = []
        for batch, data in enumerate(train_dataloader):
            subgraph, labels = data
            subgraph.set_n_initializer(dgl.init.zero_initializer)
            subgraph.set_e_initializer(dgl.init.zero_initializer)
            feats = subgraph.ndata['h'].to(device)
            labels = labels.to(device)
            if fw == 'dgl':
                model.g = subgraph
                for layer in model.layers:
                    layer.g = subgraph
                logits = model(feats.float())
            else:
                if net_class in [PGCN, PGAT, GGN]:
                    data = Data(x=feats.float(), edge_index=torch.stack(subgraph.edges()).to(device))
                else:
                    data = Data(x=feats.float(), edge_index=torch.stack(subgraph.edges()).to(device), edge_type=subgraph.edata['rel_type'].squeeze().to(device))
                logits = model(data)
            loss = loss_fcn(logits[getMaskForBatch(subgraph)], labels.float())
            optimizer.zero_grad()
            a = list(model.parameters())[0].clone()
            loss.backward()
            optimizer.step()
            b = list(model.parameters())[0].clone()
            not_learning = torch.equal(a.data, b.data)
            if not_learning:
                import sys
                print('Not learning')
                # sys.exit(1)
            else:
                pass
                # print('Diff: ', (a.data-b.data).sum())
            # print(loss.item())
            loss_list.append(loss.item())
        loss_data = np.array(loss_list).mean()
        print('Loss: {}'.format(loss_data))
        if epoch % 5 == 0:
            if epoch % 5 == 0:
                print("Epoch {:05d} | Loss: {:.4f} | Patience: {} | ".format(epoch, loss_data, cur_step), end='')
            score_list = []
            val_loss_list = []
            for batch, valid_data in enumerate(valid_dataloader):
                subgraph, labels = valid_data
                subgraph.set_n_initializer(dgl.init.zero_initializer)
                subgraph.set_e_initializer(dgl.init.zero_initializer)
                feats = subgraph.ndata['h'].to(device)
                labels = labels.to(device)
                score, val_loss = evaluate(feats.float(), model, subgraph, labels.float(), loss_fcn, fw, net_class)
                score_list.append(score)
                val_loss_list.append(val_loss)
            mean_score = np.array(score_list).mean()
            mean_val_loss = np.array(val_loss_list).mean()
            if epoch % 5 == 0:
                print("Score: {:.4f} MEAN: {:.4f} BEST: {:.4f}".format(mean_score, mean_val_loss, best_loss))
            # early stop
            if best_loss > mean_val_loss or best_loss < 0:
                print('Saving...')
                best_loss = mean_val_loss
                # Save the model
                torch.save(model.state_dict(), fw + net + '.tch')
                params = { 'loss': best_loss,
                           'net': net,
                           'fw': fw,
                           'num_layers': num_layers,
                           'num_feats': num_feats,
                           'num_hidden': num_hidden,
                           'graph_type' : graph_type,
                           'n_classes': n_classes,
                           'heads': heads,
                           'F': F.relu,
                           'in_drop': in_drop,
                           'attn_drop': attn_drop,
                           'alpha': alpha,
                           'residual': residual,
                           'num_rels': num_rels
                           }
                pickle.dump(params, open(fw + net +'.prms', 'wb'))
                cur_step = 0
            else:
                cur_step += 1
                if cur_step >= patience:
                    break
    test_score_list = []
    for batch, test_data in enumerate(test_dataloader):
        subgraph, labels = test_data
        subgraph.set_n_initializer(dgl.init.zero_initializer)
        subgraph.set_e_initializer(dgl.init.zero_initializer)
        feats = subgraph.ndata['h'].to(device)
        labels = labels.to(device)
        test_score_list.append(evaluate(feats, model, subgraph, labels.float(), loss_fcn, fw, net_class)[1])
    print("MSE for the test set {}".format(np.array(test_score_list).mean()))
    model.eval()
    return best_loss


if __name__ == '__main__':
    #list_of_tasks = pickle.load(open('LIST_OF_TASKS.pckl', 'rb'))

    #while True:
        #index = random.randrange(start=0, stop=len(list_of_tasks))
        #while list_of_tasks[index]['loss'] >= 0:
            #index += 1
        #list_of_tasks[index]['loss'] = 0
        #pickle.dump(list_of_tasks, open('LIST_OF_TASKS.pckl', 'wb'))

        #task = list_of_tasks[index]

        #if task['lr'] == 0.00001 or task['lr'] == 0.000005:
            #task['lr'] = 0.00001
            #list_of_tasks[index] = task

        #best_loss = main('socnav_training_dup.json', 'socnav_dev.json', 'socnav_test.json',
           #graph_type=task['graph_type'],
           #net=task['architecture'],
           #epochs=task['epochs'],
           #patience=task['patience'],
           #batch_size=task['batch_size'],
           #num_hidden=task['num_hidden'],
           #num_heads=task['num_heads'],
           #num_out_heads=task['num_out_heads'],
           #residual=False,
           #lr=task['lr'],
           #weight_decay=task['weight_decay'],
           #num_layers=task['num_layers'],
           #in_drop=task['in_drop'],
           #alpha=task['alpha'],
           #attn_drop=task['attn_drop'],
           #cuda=True,
           #fw=task['fw'])

        #list_of_tasks[index]['loss'] = best_loss
        #pickle.dump(list_of_tasks, open('LIST_OF_TASKS.pckl', 'wb'))


        best_loss = main('socnav_training_dup.json', 'socnav_dev.json', 'socnav_test.json',
           graph_type='1',
           net='gat',
           epochs=1000,
           patience=5,
           batch_size=294,
           num_hidden=125,
           num_heads=6,
           num_out_heads=3,
           residual=False,
           lr=0.0008,
           weight_decay=1e-04,
           num_layers=3,
           in_drop=0.0,
           alpha=0.2110715955180874,
           attn_drop=0.0,
           cuda=True,
           fw='dgl')





