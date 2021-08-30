import os
import re
import json
from collections import defaultdict

def load_texts(filename):
    res={}
    name=filename
    if "_texts.jsonl" not in filename:
        name="{}_texts.jsonl".format(filename)
    with open(name,"r", encoding='utf-8') as inn:
        for l in inn:
            tmp=json.loads(l)
            res[tmp['id']]=tmp
    return res


def load_pairs(filename):
    res=[]
    name=filename
    if "_pairs.jsonl" not in filename:
        name="{}_pairs.jsonl".format(filename)
    with open(name,"r", encoding='utf-8') as inn:
        for l in inn:
            tmp=tuple(json.loads(l))
            res.append(tmp)
    return res


def dataset_iter(dataset_dir="./res_pairs/"):
    """
    Get main dataset iterator
    :param dataset_dir: Optional[string] - path to news pairs dataset
    :return: iterator[list[tuple[dict,dict]]]
    """
    processed=set()
    files=sorted(set(re.search("(.+)(?:_pairs|_texts)",n).group(1) for n in os.listdir(dataset_dir)), key=lambda x: (len(x),x))
    for f in files:
        pairs=set(load_pairs(os.path.join(dataset_dir,f)))
        texts=load_texts(os.path.join(dataset_dir,f))
        yield [(texts[c],texts[e]) for c,e in sorted(pairs) if (c,e) not in processed]
        processed|=pairs


def get_art_summary(pair):
    """
    Convert dataset pair to absract-article pair
    :param pair: tuple[dict,dict] - dataset pair
    :return: tuple[string,string]
    """
    return (pair[0]['text'].split('\\n')[0], pair[1]['text'].replace('\\n', ' '))


def load_hals(path='./old_hals/'):
    """
    Load hallucination examples
    :param path: Optional[string] - path to files directory
    :return: dict[list] - dictionary where key is hallucination level
    """
    res=defaultdict(list)
    for h in os.listdir(path):
        level=h.split('.')[0]
        with open(os.path.join(path,h),encoding='utf-8') as f:
            for l in f:
                res[level].append(json.loads(l))
    return res