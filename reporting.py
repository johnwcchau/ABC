#%%
import pandas as pd
from pandas.core.frame import DataFrame
from model import Model, LockModel
import traceback
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


def formatnums(str):
    import re
    ints = []
    if ',' in str:
        str = str.split(',')
    else:
        str = [str]
    for s in str:
        try:
            g = re.search(r'([0-9]+)\-([0-9]+)', s)
            if g is not None:
                start = int(g.group(1))
                end = int(g.group(2))
                ints += range(start, end+1)
            else:
                ints.append(int(s))
        except Exception:
            pass
    return ints

def filter_dataset(dataset: DataFrame, criteria:str=''):
    try:
        if criteria is None: return dataset
        criterias = criteria.split("&&")
        for criteria in criterias:
            criteria = criteria.strip()
            if criteria.startswith('ts:'):
                criteria = criteria.replace("ts:", "").split(",")
                for c in criteria:
                    if c[0] == ">":
                        c = pd.to_datetime(c[1:])
                        dataset = dataset[dataset.ts>c]
                    elif c[0] == "<":
                        c = pd.to_datetime(c[1:])
                        dataset = dataset[dataset.ts<c]
                    else:
                        c = pd.to_datetime(c)
                        dataset = dataset[dataset.ts==c]
            elif criteria.startswith('ref:'):
                dataset = dataset[dataset.ref==criteria.replace('ref:', '').strip()]
            elif criteria.startswith('predict:'):
                tgt = formatnums(criteria.replace("predict:", "").strip())
                if len(tgt) == 0:
                    continue
                else:
                    dataset = dataset[dataset.predict.isin(tgt)]
            elif criteria.startswith('target:'):
                tgt = formatnums(criteria.replace("target:", "").strip())
                if len(tgt) == 0:
                    continue
                else:
                    dataset = dataset[dataset.target.isin(tgt)]
            elif criteria != '':
                dataset = dataset[dataset.text.str.lower().str.contains(criteria.lower())]
    except Exception:
        traceback.print_exc()
    return dataset

def search_dataset(model: Model, set:str='train', criteria:str='', start:int=0, count:int=20):
    dataset = model.devset if set == 'dev' else model.trainset
    total = dataset.shape[0]
    dataset = filter_dataset(dataset, criteria)

    filtered = dataset.shape[0]
    dataset = dataset.iloc[start:start+count][["text", "ts", "ref", "target", "predict"]]
    dataset = dataset.fillna('')
    dataset['id'] = dataset.index
    return {
        'recordsTotal': total,
        'recordsFiltered': filtered,
        'data': dataset.to_dict(orient='records')
    }

@LockModel
def get_group_info(model: Model, **params):
    ts = model.group_summaries
    if ts is None:
        model.log.invalid("Use 'Generate Stats' first")
        return
    if "group" not in params or params["group"] < 0:
        model.log.invalid()
        return
    #
    # find similar groups
    #
    group = int(params["group"])
    simscore = euclidean_distances(ts["mean"].to_list(), Y=[ts.loc[group]["mean"]]).reshape(-1)
    similarities = np.argsort(simscore)[1:6]
    similar = []
    for s in similarities:
        t = ts.loc[s]
        similar.append({
            "group": int(t["group"]),
            "distance": float(simscore[s]),
            "score": float(simscore[s]) / model.group_stats["max_distance"],
            "repr_text": t["repr_text"],
        })
    group = ts.loc[group].to_dict()
    del group["mean"]
    group["count"] = int(group["count"])
    group["group"] = int(group["group"])
    group["distance"] = float(group["distance"])
    group["rel_size"] = float(group["rel_size"])
    params = {
        "group": group,
        "neighbours": similar,
    }
    model.log.finished(None, param = params)

@LockModel
def text_similarity(model: Model, **params):
    #
    # Silent UI call, no message return
    #
    if model.group_summaries is None:
        model.log.invalid(None)
        return
    if "x" not in params and "y" not in params:
        model.log.invalid(None)
        return
    
    dataset = model.devset if 'set' in params and params['set']=='dev' else model.trainset
    x = dataset[dataset.index.isin(params['x'])]['umap'].to_list()
    y = dataset.iloc[params['y']]['umap']
    result = (1 - (euclidean_distances(x, Y=[y]) / model.group_stats["max_distance"]).reshape(-1)).tolist()
    result = dict(zip(params["x"], result))
    model.log.finished(param = {
        "similarities": result
    })

@LockModel
def text_summary(model: Model, **params):
    def get(dataset):
        #dataset = model.devset if 'set' in params and params['set']=='dev' else model.trainset
        hidefiltered = True if 'hidefiltered' in params and params['hidefiltered'] else False
        filter = params["filter"] if 'filter' in params else None
        filtered_dataset = filter_dataset(dataset, filter)
        if hidefiltered and filter:
            dataset = filtered_dataset
        dataset = dataset[dataset.predict.notna()]
        if dataset.shape[0] > 20000:
            per_group = 20000 // model.group_stats["num_group"]
            if per_group < 3: per_group = 3
            dataset = dataset.groupby('predict').head(per_group)
        dataset = dataset[['text', 'x', 'y', 
            #'z', 
            'ref', 'predict']].copy()
        dataset['text'] = dataset['text'].str.slice(0, 80)
        dataset.reset_index(inplace=True)
        
        dataset.rename(columns={"index":"id"}, inplace=True)
        if filter and not hidefiltered:
            not_filtered = filtered_dataset.reset_index()["index"].to_list()
            dataset["filtered"] = True
            dataset.loc[dataset.id.isin(not_filtered), "filtered"] = False
        else:
            dataset["filtered"] = False
        return (filtered_dataset, dataset.fillna("").to_dict(orient='records'))
    train_filtered, train_result = get(model.trainset)
    train_filtered = train_filtered.groupby("predict").first().reset_index()["predict"]
    model.log.finished(param = {
        'train': train_result,
        'dev': get(model.devset)[1],
        'groups': train_filtered.to_list(),
    })

@LockModel
def group_search(model: Model, **params):
    dataset = model.devset if 'set' in params and params['set']=='dev' else model.trainset
    if 'filter' in params and params['filter'] != '':
        dataset = filter_dataset(dataset, params["filter"])
    dataset = dataset.groupby("predict").first().reset_index()["predict"]
    model.log.finished(param = {
        'groups': dataset.to_list()
    })


@LockModel
def group_over_time(model: Model, **kwargs):
    try:
        model.log.working("Calculating group trends...")
        dataset = model.trainset[["ts", "predict"]].copy()
        dataset = dataset.append(model.devset[["ts", "predict"]].copy()).drop_duplicates()
        uniquets = dataset.ts.unique().shape[0]
        if uniquets > 100:
            dataset["tsbin"] = pd.cut(dataset.ts, bins=100)
            dataset["tsbin"] = dataset.apply(lambda row: row.tsbin.left, 1)
        else:
            dataset["tsbin"] = dataset["ts"]
        counts = dataset.sort_values(["tsbin", "predict"]).groupby(by=["tsbin", "predict"], as_index=False).count()
        result = [{"idx": i, "text": model.group_summaries.loc[i, "repr_text"][0:80] if i > -1 else "", "x": [], "y": []} for i in range(-1, model.group_summaries.shape[0])]

        def toresult(row):
            result[row.predict+1]["x"].append(row.tsbin)
            result[row.predict+1]["y"].append(row.ts)
        counts.apply(toresult, axis=1)
        model.log.finished("Group counts over time ready", param = {"counts": result})
    except Exception:
        traceback.print_exc()
        model.log.error("Error calculating group trends")
        return
# %%
