import pickle
from random import choice, randrange, uniform



fw_net_map = { 0:  ('dgl', 'gcn'),
               1:  ('dgl', 'gat'),
               2:  ('pg',  'gcn'),
               3:  ('pg',  'gat'),
               4:  ('pg',  'rgcn'),
               5:  ('pg',  'ggn'),
               6:  ('pg',  'rgat'),
               7:  ('pg',  'rgat3')
             }


best_by_option = {}
try:
    list_of_tasks = pickle.load(open('LIST_OF_TASKS.pckl', 'rb'))
    best_loss = -1
    best = None
    best_index = None
    pending, failed, done = 0, 0, 0
    for index in range(len(list_of_tasks)):
        # Get info
        selected = list_of_tasks[index]
        loss, fw, architecture = selected['loss'], selected['fw'], selected['architecture']
        # Handle: pending/failed/done
        if loss < -0.6:
            pending += 1
        elif loss <= 0.:
            failed += 1
        else:
            done += 1
        # Handle overall best
        if (loss < best_loss and loss > 0.) or best_loss <= 0:
            best_loss = loss
            best = list_of_tasks[index]
            best_index = index
        # Handle best by type
        if (fw,architecture) in best_by_option:
            if loss > 0:
                if loss < best_by_option[(fw,architecture)]['loss']:
                    best_by_option[(fw,architecture)] = selected
        elif loss>0:
            best_by_option[(fw,architecture)] = selected

    print('BEST BY OPTION')
    for k in best_by_option:
        print(k)
        print(best_by_option[k])
        print('')
    print('Pending:', pending)
    print('Failed:', failed)
    print('Done:', done)
    print('Best result:', best)
    print('Best index:', best_index)
except FileNotFoundError:
    list_of_tasks = []
    for i in range(5000):
        fw, architecture = fw_net_map[randrange(len(fw_net_map))]
        if architecture in ['rgcn', 'rgat', 'rgat2', 'rgat3']:
            graph_type = choice(['relational', '4', 'dani1'])
            graph_type = 'dani1'
        else:
            graph_type = '4'
            graph_type = 'dani1'
        hyperparameters = {
            'fw': fw,
            'architecture': architecture,
            'graph_type': graph_type,
            'epochs': 1000, # 1000
            'patience': 5, # 5
            'batch_size': randrange(start=50, stop=300), # start=100, stop=1500
            'num_hidden': randrange(start=50, stop=320), # start=50, stop=320)
            'num_heads': randrange(start=2, stop=9),
            'num_out_heads': randrange(start=2, stop=9),
            'lr': choice([0.001, 0.0005, 0.0001, 0.00005, 0.000025, 0.00001, 0.000005]),
            'weight_decay': choice([0.00001, 0.000005, 0.000001, 0.0000005, 0.0000001, 0.0]),
            'num_layers': randrange(start=2, stop=8),
            'in_drop': choice([0.00001, 0.000001, 0.0, 0.0, 0.0, 0.0]),
            'alpha': uniform(0.1, 0.3),
            'attn_drop': 0.0,
            'loss': -1.
        }

        list_of_tasks.append(hyperparameters)
        pickle.dump(list_of_tasks, open('LIST_OF_TASKS.pckl', 'wb'))





