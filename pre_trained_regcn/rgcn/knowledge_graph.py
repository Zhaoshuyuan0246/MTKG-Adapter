""" Knowledge graph dataset for Relational-GCN
Code adapted from authors' implementation of Relational-GCN
https://github.com/tkipf/relational-gcn
https://github.com/MichSchli/RelationPrediction
"""

from __future__ import absolute_import
from __future__ import print_function

import gzip
import os
from collections import Counter

import numpy as np
import pandas as pd
import rdflib as rdf
import scipy.sparse as sp
# from dgl.data.utils import download, extract_archive, get_download_dir, _get_dgl_url
import sys
import json
from collections import defaultdict
import glob
from PIL import Image
import torch
from torchvision import transforms

np.random.seed(123)


class RGCNLinkDataset(object):
    def __init__(self, name, dir=None, image_size=224):
        self.name = name
        self.image_size = image_size
        self.image_transform = self._get_image_transform()
        
        if dir:
            self.dir = dir
            self.dir = os.path.join(self.dir, self.name)
        else:
            self.dir = get_download_dir()
            tgz_path = os.path.join(self.dir, '{}.tar.gz'.format(self.name))
            download(_downlaod_prefix + '{}.tgz'.format(self.name), tgz_path)
            self.dir = os.path.join(self.dir, self.name)
            extract_archive(tgz_path, self.dir)
        print(self.dir)

    def _get_image_transform(self):
        """Image preprocessing transform."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def _load_entity_images(self):
        """Load entity image data."""
        picture_dir = os.path.join(self.dir, 'picture')
        entity_images = {}
        
        if not os.path.exists(picture_dir):
            print(f"警告: 图像目录 {picture_dir} 不存在")
            return entity_images
        
        # iterate over all entity folders
        entity_folders = glob.glob(os.path.join(picture_dir, '*'))
        
        for entity_folder in entity_folders:
            if os.path.isdir(entity_folder):
                entity_name = os.path.basename(entity_folder)
                
                # look up the corresponding entity id in entity_dict
                entity_id = None
                for eid, ename in self.entity_dict.items():
                    if ename.lower() == entity_name.lower():
                        entity_id = int(eid)
                        break
                
                if entity_id is not None:
                    # load all images of this entity
                    image_files = glob.glob(os.path.join(entity_folder, '*.jpg')) + \
                                 glob.glob(os.path.join(entity_folder, '*.png')) + \
                                 glob.glob(os.path.join(entity_folder, '*.jpeg'))
                    
                    entity_images[entity_id] = []
                    for img_path in image_files:
                        try:
                            # store the image path; load it when needed
                            entity_images[entity_id].append(img_path)
                        except Exception as e:
                            print(f"加载图像失败 {img_path}: {e}")
        
        print(f"成功加载 {len(entity_images)} 个实体的图像数据")
        return entity_images

    def _load_single_image(self, image_path):
        """Load a single image."""
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.image_transform(image)
            return image
        except Exception as e:
            print(f"加载图像失败 {image_path}: {e}")
            # return a blank image as a placeholder
            return torch.zeros(3, self.image_size, self.image_size)

    def get_entity_images(self, entity_id, max_images=5):
        """Return the image data of a given entity."""
        if hasattr(self, 'entity_images') and entity_id in self.entity_images:
            image_paths = self.entity_images[entity_id][:max_images]
            images = [self._load_single_image(path) for path in image_paths]
            
            # pad with blank images if there are not enough images
            while len(images) < max_images:
                images.append(torch.zeros(3, self.image_size, self.image_size))
            
            return torch.stack(images)  # [max_images, 3, H, W]
        else:
            # return a blank image
            return torch.zeros(max_images, 3, self.image_size, self.image_size)

    def get_triplet_images(self, head_id, tail_id, max_images_per_entity=3):
        """Return the triplet images (head-entity and tail-entity images)."""
        head_images = self.get_entity_images(head_id, max_images_per_entity)
        tail_images = self.get_entity_images(tail_id, max_images_per_entity)
        
        # merge head-entity and tail-entity images
        triplet_images = torch.cat([head_images, tail_images], dim=0)  # [2*max_images_per_entity, 3, H, W]
        return triplet_images

    def load(self, load_time=True):
        entity_path = os.path.join(self.dir, 'entity2id.txt')
        relation_path = os.path.join(self.dir, 'relation2id.txt')
        train_path = os.path.join(self.dir, 'train.txt')
        test_path = os.path.join(self.dir, 'test.txt')
        
        entity_dict = _read_dictionary(entity_path)
        relation_dict = _read_dictionary(relation_path)
        
        self.entity_dict = entity_dict
        self.relation_dict = relation_dict
        self.train = np.array(_read_triplets_as_list(train_path, entity_dict, relation_dict, load_time))
        
        # load text data
        with open(os.path.join(self.dir, "train_text.json"), "r", encoding="utf-8") as f:
            self.train_text = json.load(f)
        with open(os.path.join(self.dir, "test_text.json"), "r", encoding="utf-8") as f:
            self.test_text = json.load(f)

        # load pretrained-model scores (if present)
        pretrained_score_path = "top100/Wiki_top100_predictions.json"
        if os.path.exists(pretrained_score_path):
            with open(pretrained_score_path, "r", encoding="utf-8") as f:
                self.pretrained_test_scores = json.load(f)
        else:
            self.pretrained_test_scores = {}



        self.test_tail_sets = build_all_ans(train_path, test_path)

        # load image data
        self.entity_images = self._load_entity_images()
        #self.entity_images = {}  # initialize to empty dict first, then load images lazily}
        
        self.test_tail_sets = build_all_ans(train_path, test_path)
        self.test = np.array(_read_triplets_as_list(test_path, entity_dict, relation_dict, load_time))
        
        # load the validation set (if present)
        val_path = os.path.join(self.dir, 'val.txt')
        if os.path.exists(val_path):
            self.valid = np.array(_read_triplets_as_list(val_path, entity_dict, relation_dict, load_time))
        else:
            self.valid = np.array([])
        
        self.num_nodes = len(entity_dict)
        print("# Sanity Check:  entities: {}".format(self.num_nodes))
        self.num_rels = len(relation_dict)
        print("# Sanity Check:  relations: {}".format(self.num_rels))
        #train|test|valid
        print("# Sanity Check:  edges: {}".format(len(self.train)))
        #print("# Sanity Check:  edges: {}".format(len(self.test)))
        print("# Sanity Check:  images: {} entities have images".format(len(self.entity_images)))

    def get_sample_with_images(self, idx, split='train', max_images=5):
        """Return a sample with image data."""
        if split == 'train':
            triplets = self.train
            text_data = self.train_text
        elif split == 'valid':
            triplets = self.valid
            text_data = self.val_text
        else:  # test
            triplets = self.test
            text_data = self.test_text
        
        if idx >= len(triplets):
            return None
        
        head, rel, tail, timestamp = triplets[idx]
        
        # build the text key (used to look up the text description)
        text_key = f"{self.entity_dict[str(head)]}\t{self.relation_dict[str(rel)]}\t{self.entity_dict[str(tail)]}\t{timestamp}"
        
        # get the text description
        text_description = text_data.get(text_key, "")
        
        # get image data
        head_images = self.get_entity_images(head, max_images)
        tail_images = self.get_entity_images(tail, max_images)
        
        return {
            'head_id': head,
            'relation_id': rel,
            'tail_id': tail,
            'timestamp': timestamp,
            'text_description': text_description,
            'head_images': head_images,      # [max_images, 3, H, W]
            'tail_images': tail_images,      # [max_images, 3, H, W]
            'head_entity_name': self.entity_dict[str(head)],
            'tail_entity_name': self.entity_dict[str(tail)],
            'relation_name': self.relation_dict[str(rel)]
        }


def load_entity(dataset, bfs_level, relabel):
    data = RGCNEntityDataset(dataset)
    data.load(bfs_level, relabel)
    return data


def load_link(dataset):
    data = RGCNLinkDataset(dataset)
    data.load()
    return data


def load_from_local(dir, dataset):
    data = RGCNLinkDataset(dataset, dir)# create an RGCNLinkDataset instance
    data.load()
    return data


def _sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return sp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def _get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors."""
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(sp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors


def _bfs_relational(adj, roots):
    """
    BFS for graphs with multiple edge types. Returns list of level sets.
    Each entry in list corresponds to relation specified by adj_list.
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        yield next_lvl

        current_lvl = set.union(next_lvl)


class RDFReader(object):
    __graph = None
    __freq = {}

    def __init__(self, file):

        self.__graph = rdf.Graph()

        if file.endswith('nt.gz'):
            with gzip.open(file, 'rb') as f:
                self.__graph.parse(file=f, format='nt')
        else:
            self.__graph.parse(file, format=rdf.util.guess_format(file))

        # See http://rdflib.readthedocs.io for the rdflib documentation

        self.__freq = Counter(self.__graph.predicates())

        print("Graph loaded, frequencies counted.")

    def triples(self, relation=None):
        for s, p, o in self.__graph.triples((None, relation, None)):
            yield s, p, o

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__graph.destroy("store")
        self.__graph.close(True)

    def subjectSet(self):
        return set(self.__graph.subjects())

    def objectSet(self):
        return set(self.__graph.objects())

    def relationList(self):
        """
        Returns a list of relations, ordered descending by frequency
        :return:
        """
        res = list(set(self.__graph.predicates()))
        res.sort(key=lambda rel: - self.freq(rel))
        return res

    def __len__(self):
        return len(self.__graph)

    def freq(self, rel):
        if rel not in self.__freq:
            return 0
        return self.__freq[rel]


def _load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'], dtype=np.float32)


def _save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def _load_data(dataset_str='aifb', dataset_path=None):
    """

    :param dataset_str:
    :param rel_layers:
    :param limit: If > 0, will only load this many adj. matrices
        All adjacencies are preloaded and saved to disk,
        but only a limited a then restored to memory.
    :return:
    """

    print('Loading dataset', dataset_str)

    graph_file = os.path.join(dataset_path, '{}_stripped.nt.gz'.format(dataset_str))
    task_file = os.path.join(dataset_path, 'completeDataset.tsv')
    train_file = os.path.join(dataset_path, 'trainingSet.tsv')
    test_file = os.path.join(dataset_path, 'testSet.tsv')
    if dataset_str == 'am':
        label_header = 'label_category'
        nodes_header = 'proxy'
    elif dataset_str == 'aifb':
        label_header = 'label_affiliation'
        nodes_header = 'person'
    elif dataset_str == 'mutag':
        label_header = 'label_mutagenic'
        nodes_header = 'bond'
    elif dataset_str == 'bgs':
        label_header = 'label_lithogenesis'
        nodes_header = 'rock'
    else:
        raise NameError('Dataset name not recognized: ' + dataset_str)

    edge_file = os.path.join(dataset_path, 'edges.npz')
    labels_file = os.path.join(dataset_path, 'labels.npz')
    train_idx_file = os.path.join(dataset_path, 'train_idx.npy')
    test_idx_file = os.path.join(dataset_path, 'test_idx.npy')

    if os.path.isfile(edge_file) and os.path.isfile(labels_file) and \
            os.path.isfile(train_idx_file) and os.path.isfile(test_idx_file):

        # load precomputed adjacency matrix and labels
        all_edges = np.load(edge_file)
        num_node = all_edges['n'].item()
        edge_list = all_edges['edges']
        num_rel = all_edges['nrel'].item()

        print('Number of nodes: ', num_node)
        print('Number of edges: ', len(edge_list))
        print('Number of relations: ', num_rel)

        labels = _load_sparse_csr(labels_file)
        labeled_nodes_idx = list(labels.nonzero()[0])

        print('Number of classes: ', labels.shape[1])

        train_idx = np.load(train_idx_file)
        test_idx = np.load(test_idx_file)

    else:

        # loading labels of nodes
        labels_df = pd.read_csv(task_file, sep='\t', encoding='utf-8')
        labels_train_df = pd.read_csv(train_file, sep='\t', encoding='utf8')
        labels_test_df = pd.read_csv(test_file, sep='\t', encoding='utf8')

        with RDFReader(graph_file) as reader:

            relations = reader.relationList()
            subjects = reader.subjectSet()
            objects = reader.objectSet()

            nodes = list(subjects.union(objects))
            num_node = len(nodes)
            num_rel = len(relations)
            num_rel = 2 * num_rel + 1 # +1 is for self-relation

            assert num_node < np.iinfo(np.int32).max
            print('Number of nodes: ', num_node)
            print('Number of relations: ', num_rel)

            relations_dict = {rel: i for i, rel in enumerate(list(relations))}
            nodes_dict = {node: i for i, node in enumerate(nodes)}

            edge_list = []
            # self relation
            for i in range(num_node):
                edge_list.append((i, i, 0))

            for i, (s, p, o) in enumerate(reader.triples()):
                src = nodes_dict[s]
                dst = nodes_dict[o]
                assert src < num_node and dst < num_node
                rel = relations_dict[p]
                # relation id 0 is self-relation, so others should start with 1
                edge_list.append((src, dst, 2 * rel + 1))
                # reverse relation
                edge_list.append((dst, src, 2 * rel + 2))

            # sort indices by destination
            edge_list = sorted(edge_list, key=lambda x: (x[1], x[0], x[2]))
            edge_list = np.array(edge_list, dtype=np.int)
            print('Number of edges: ', len(edge_list))

            np.savez(edge_file, edges=edge_list, n=np.array(num_node), nrel=np.array(num_rel))

        nodes_u_dict = {np.unicode(to_unicode(key)): val for key, val in
                        nodes_dict.items()}

        labels_set = set(labels_df[label_header].values.tolist())
        labels_dict = {lab: i for i, lab in enumerate(list(labels_set))}

        print('{} classes: {}'.format(len(labels_set), labels_set))

        labels = sp.lil_matrix((num_node, len(labels_set)))
        labeled_nodes_idx = []

        print('Loading training set')

        train_idx = []
        train_names = []
        for nod, lab in zip(labels_train_df[nodes_header].values,
                            labels_train_df[label_header].values):
            nod = np.unicode(to_unicode(nod))  
            if nod in nodes_u_dict:
                labeled_nodes_idx.append(nodes_u_dict[nod])
                label_idx = labels_dict[lab]
                labels[labeled_nodes_idx[-1], label_idx] = 1
                train_idx.append(nodes_u_dict[nod])
                train_names.append(nod)
            else:
                print(u'Node not in dictionary, skipped: ',
                      nod.encode('utf-8', errors='replace'))

        print('Loading test set')

        test_idx = []
        test_names = []
        for nod, lab in zip(labels_test_df[nodes_header].values,
                            labels_test_df[label_header].values):
            nod = np.unicode(to_unicode(nod))
            if nod in nodes_u_dict:
                labeled_nodes_idx.append(nodes_u_dict[nod])
                label_idx = labels_dict[lab]
                labels[labeled_nodes_idx[-1], label_idx] = 1
                test_idx.append(nodes_u_dict[nod])
                test_names.append(nod)
            else:
                print(u'Node not in dictionary, skipped: ',
                      nod.encode('utf-8', errors='replace'))

        labeled_nodes_idx = sorted(labeled_nodes_idx)
        labels = labels.tocsr()
        print('Number of classes: ', labels.shape[1])

        _save_sparse_csr(labels_file, labels)

        np.save(train_idx_file, train_idx)
        np.save(test_idx_file, test_idx)

        # np.save(train_names_file, train_names)
        # np.save(test_names_file, test_names)

        # pkl.dump(relations_dict, open(rel_dict_file, 'wb'))

    # end if

    return num_node, edge_list, num_rel, labels, labeled_nodes_idx, train_idx, test_idx


def to_unicode(input):
    # FIXME (lingfan): not sure about python 2 and 3 str compatibility
    return str(input)
    """ lingfan: comment out for now
    if isinstance(input, unicode):
        return input
    elif isinstance(input, str):
        return input.decode('utf-8', errors='replace')
    return str(input).decode('utf-8', errors='replace')
    """


def _read_dictionary(filename):
    d = {}
    with open(filename, 'r') as f:  # changed to 'r' mode; no need to read/write
        for line in f:
            line = line.strip().split()  # split on whitespace (default split handles any whitespace)
            if len(line) >= 2:
                d[int(line[0])] = ' '.join(line[1:])  # id is in the first column, name in the rest
    return d


def _read_triplets(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:          # skip empty lines
                continue
            parts = line.split()  # split on whitespace by default (spaces, tabs, etc.)
            if len(parts) == 4:
                yield [int(x) for x in parts]


def _read_triplets_as_list(filename, entity_dict, relation_dict, load_time):
    l = []
    for triplet in _read_triplets(filename):
        s = int(triplet[0])
        r = int(triplet[1])
        o = int(triplet[2])
        if load_time:
            st = int(triplet[3])
            l.append([s, r, o, st])
        else:
            l.append([s, r, o])
    return l

def generate_text_from_triples(triples, entity_dict, relation_dict):

    texts = []
    
    # iterate over each triple and build the corresponding text
    for triple in triples:
        src, rel, dst, de = triple # corresponds to: s, r, o, st

        # get the readable entity and relation names
        src_name = entity_dict.get(src, f"Entity_{src}")
        dst_name = entity_dict.get(dst, f"Entity_{dst}")
        rel_name = relation_dict.get(rel, f"Relation_{rel}")
        
        prompt = ""
        prompt += f"Question: \nAt {de}, {src_name} has a {rel_name} relationship with which entity?\n"
        prompt += f"Extracted Quadruple: {de}: [{src_name}, {rel_name}, [MISSING ENTITY ID].[MISSING ENTITY]]\n"
        prompt += f"Historical Context:\n"
        prompt += "None\n"
        prompt += "Answer:\n"

        text = {
            # 'instruction': prompt + f"At {de}, {src_name} has a {rel_name} relationship with which entity?\n",
            'instruction': prompt,
            'response': f"{dst}.{dst_name}"
        }
        texts.append(text)
    
    return texts

def build_all_ans(train_path, test_path):
    # build the nested answer dict all_ans[h][r] = set(tail)
    all_ans = defaultdict(lambda: defaultdict(set))

    # read train.txt data and fill all_ans
    with open(train_path, 'r') as f:
        for line in f:
            h, r, t, _ = map(int, line.strip().split())
            all_ans[h][r].add(t)

    # for each line in test.txt, collect the tail set of its (h, r)
    test_ans = []

    with open(test_path, 'r') as f:
        for line in f:
            h, r, _, _ = map(int, line.strip().split())
            tails = all_ans[h][r] if r in all_ans[h] else set()
            test_ans.append(tails)

    return test_ans