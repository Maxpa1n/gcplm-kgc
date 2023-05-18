import json
def load_triples(path,lang):
    triples = []
    with open(path, 'r', encoding="utf8") as fh:
        for line in fh.readlines():
            k = line.strip().split('\t')
            if len(k)!=4:
                print(k)
                continue
            e1, r, e2, lan = k
            if lan == lang or (lang =="any" and lan!="en"):
            # if 1==1:
                triples.append((e1, r, e2, lan))
    return triples


def load_entities(path):
    entities_list = []
    with open(path, 'r', encoding="utf8") as fh:
        for line in fh.readlines():
            lan = json.loads(line.strip())
            e1 = lan["en"]
            e2 = lan["th"]
            entities_list.append((e1, e2))
    return entities_list