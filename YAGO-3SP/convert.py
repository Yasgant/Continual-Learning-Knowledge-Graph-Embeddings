train_edges = []
valid_edges = []
test_edges = []
num = 3
for i in range(1, num+1):
    ent, rel = {}, {}
    with open(f'snapshot{i}/entity2id.txt', 'r') as f:
        f.readline()
        for line in f.readlines():
            line = line.strip().split()
            ent[line[1]] = line[0]
    with open(f'snapshot{i}/relation2id.txt', 'r') as f:
        f.readline()
        for line in f.readlines():
            line = line.strip().split()
            rel[line[1]] = line[0]
    edges = []
    with open(f'snapshot{i}/train2id.txt', 'r') as f:
        f.readline()
        for line in f.readlines():
            line = line.strip().split()
            edges.append((ent[line[0]], rel[line[2]], ent[line[1]]))
    train_edges.append(edges)
    edges = []
    with open(f'snapshot{i}/valid2id.txt', 'r') as f:
        f.readline()
        for line in f.readlines():
            line = line.strip().split()
            edges.append((ent[line[0]], rel[line[2]], ent[line[1]]))
    valid_edges.append(edges)
    edges = []
    with open(f'snapshot{i}/test2id.txt', 'r') as f:
        f.readline()
        for line in f.readlines():
            line = line.strip().split()
            edges.append((ent[line[0]], rel[line[2]], ent[line[1]]))
    test_edges.append(edges)
convert_train = [[] for _ in range(num)]
convert_valid = [[] for _ in range(num)]
convert_test = [[] for _ in range(num)]
ent, rel = {}, {}
def get_ent(e):
    if e not in ent:
        ent[e] = len(ent)
    return ent[e]
def get_rel(r):
    if r not in rel:
        rel[r] = len(rel)
    return rel[r]
for i in range(3):
    for edge in train_edges[i]:
        convert_train[i].append((get_ent(edge[0]), get_rel(edge[1]), get_ent(edge[2])))
    for edge in valid_edges[i]:
        convert_valid[i].append((get_ent(edge[0]), get_rel(edge[1]), get_ent(edge[2])))
    for edge in test_edges[i]:
        convert_test[i].append((get_ent(edge[0]), get_rel(edge[1]), get_ent(edge[2])))
def dif(edges1, edges2):
    add, rem = [], []
    edgeset1 = set(edges1)
    edgeset2 = set(edges2)
    for edge in edges2:
        if edge not in edgeset1:
            add.append(edge)
    
    for edge in edges1:
        if edge not in edgeset2:
            rem.append(edge)
    return add, rem

for i in range(num):
    with open(f'train{i}.csv', 'w') as f:
        for edge in convert_train[i]:
            f.write(f'{edge[0]},{edge[1]},{edge[2]}\n')
    with open(f'valid{i}.csv', 'w') as f:
        for edge in convert_valid[i]:
            f.write(f'{edge[0]},{edge[1]},{edge[2]}\n')
    with open(f'test{i}.csv', 'w') as f:
        for edge in convert_test[i]:
            f.write(f'{edge[0]},{edge[1]},{edge[2]}\n')
    if i > 0:
        add, rem = dif(convert_train[i-1], convert_train[i])
        with open(f'dif{i}.csv', 'w') as f:
            for edge in add:
                f.write(f'add,{edge[0]},{edge[1]},{edge[2]}\n')
            for edge in rem:
                f.write(f'rem,{edge[0]},{edge[1]},{edge[2]}\n')

print(f'nentity: {len(ent)}, nrelation: {len(rel)}')