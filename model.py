# %%
import pandas as pd
import numpy as np
import traceback
import sqlite3
import joblib
import io
import os
from sklearn.metrics.pairwise import euclidean_distances

from util import *


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    try:
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)
    except ValueError:
        try:
            return int(text)
        except Exception:
            try:
                return float(text)
            except Exception:
                return text

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

def clean_file(_type):
    for root, _, files in os.walk("saves"):
        for name in files:
            if _type in name:
                os.remove(os.path.join(root, name))

def LockModel(fn:callable):
    def wrapper(model, *a, **k):
        result = None
        if model.busy:
            model.log.busy()
            return
        try:
            model.busy = True
            result = fn(model, *a, **k)
        except Exception:
            traceback.print_exc()
            model.log.error("Unhandled server error")
        model.busy = False
        return result
    return wrapper

class Model:

    def __init__(self, name:str=None):
        if name is None:
            from datetime import datetime
            now = datetime.now()
            name = now.strftime("model-%Y_%m_%d_%H_%M_%S")
        self.name = name
        self.log = Logger()
        self.trainset = Model.__newset()
        self.devset = Model.__newset()
        self.adhocset = None
        self.busy = False
        self.embedding_model = None
        self.reduction_model = None
        self.clustering_model = None
        self.projection_model = None
        self.model_params = {}
        self.group_stats = None
        self.group_summaries = None
        self.group_tree = None
    
    @classmethod
    def __newset(cls):
        df = pd.DataFrame(None, columns=['text', 'embedding', 'umap', 'x', 'y', 'ts', 'ref', 'target', 'predict', 'distance'])
        df["x"] = df["x"].astype(int)
        df["y"] = df["y"].astype(int)
        df["predict"] = df["predict"].astype(int)
        df["distance"] = df["distance"].astype(int)
        return df

    def set_client(self, ws):
        self.log.ws = ws
    
    def rename(self, name:str=None, **kwargs):
        if name is None:
            from datetime import datetime
            now = datetime.now()
            name = now.strftime("model-%Y_%m_%d_%H_%M_%S")
        self.name = name
        self.log.finished()

    @LockModel
    def clean_dataset(self, **param):
        if "set" not in param:
            self.log.invalid("Must specify which set to clean")
        if param["set"] == "dev":
            self.devset = Model.__newset()
        elif param["set"] == "train":
            self.trainset = Model.__newset()
            self.group_stats = None
            self.group_summaries = None
            self.group_tree = None
        self.log.finished("Datasets cleaned")
    
    def __delete_rows(self, **param):
        if "train" in param:
            self.trainset.drop(param["train"], inplace=True)
            self.trainset.reset_index(drop=True, inplace=True)
        if "dev" in param:
            self.devset.drop(param["dev"], inplace=True)
            self.devset.reset_index(drop=True, inplace=True)

    @LockModel
    def delete_rows(self, **param):
        try:
            self.__delete_rows(**param)
            self.log.finished("Rows deleted")
        except Exception:
            traceback.print_exc()
            self.log.error("Failed deleting rows")

    @LockModel
    def copy_rows(self, **param):
        totrain = True if 'to' in param and param['to'] == 'train' else False
        src = self.devset if totrain else self.trainset
        ismove = 'type' in param and param['type']=='move'
        if ismove and not self.__can_modify_trainset():
            self.log.error("Cannot modify trainset after model is trained")
            return
        try:
            df = src.loc[param['rows']].copy()
            if totrain:
                self.trainset = self.trainset.append(df).reset_index(drop=True)
            else:
                self.devset = self.devset.append(df).reset_index(drop=True)
            if ismove:
                self.__delete_rows(**{'dev' if totrain else 'train': param['rows']})
                #src.drop(index=param['rows'], inplace=True)
            self.log.finished("%s %d rows from %s to %s" % (
                'Moved' if ismove else 'Copied',
                df.shape[0],
                'devset' if totrain else 'trainset',
                'trainset' if totrain else 'devset'))
        except Exception:
            traceback.print_exc()
            self.log.error("Failed copying rows")

    def __reset_columns(self, **param):
        try:
            isdevset = True if 'set' in param and param['set'] == 'dev' else False
            dataset = self.devset if isdevset else self.trainset

            for col in ["embedding", "umap", 'x', 'y', 'predict']:
                if col in param and param[col]:
                    dataset[col] = np.nan
                    if col in ['embedding', 'umap']:
                        dataset[col] = dataset[col].astype(object)
                    if col == 'predict' and isdevset:
                        dataset['distance'] = np.nan
            return True
        except Exception:
            traceback.print_exc()
            self.log.error("Failed resetting columns")
            return False

    @LockModel
    def reset_columns(self, **param):
        if self.__reset_columns(**param):
            self.log.finished("Columns resetted")

    def __create_embedding_model(self, **param):
        log = self.log
        try:
            log.working("Creating embedding model...")
            if "type" in param: del param["type"]
            from sentence_transformers import SentenceTransformer
            with LogRedirect(self.log):
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                log.working("Embedding to be done on %s" % device)
                self.embedding_model = SentenceTransformer(param["model"], device=device)
            return True
        except Exception:
            traceback.print_exc()
            log.error("Embedding model creation failed")
            return False

    def __create_reduction_model(self, **param):
        log = self.log
        try:
            if param and 'verbose' in param:
                del param['verbose']
            log.working("Creating reduction model...")
            if "type" in param: del param["type"]
            from umap import UMAP
            self.reduction_model = UMAP(verbose=True, **param)
            return True
        except Exception:
            traceback.print_exc()
            log.error("Reduction model creation failed")
            return False

    def __create_clustering_model(self, **param):
        log = self.log
        try:
            if param and 'prediction_data' in param:
                del param['prediction_data']
            log.working("Creating clustering model...")
            if "type" in param: del param["type"]
            from hdbscan import HDBSCAN
            self.clustering_model = HDBSCAN(prediction_data=True, **param)
            return True
        except Exception:
            traceback.print_exc()
            log.error("Clustering model creation failed")
            return False
    
    def __create_projection_model(self, **param):
        log = self.log
        try:
            log.working("Creating projection model...")
            from umap import UMAP
            self.projection_model = UMAP(metric="euclidean", n_components=2, verbose=True)
            return True
        except Exception:
            traceback.print_exc()
            log.error("Projection model creation failed")
            return False

    @LockModel
    def create_model(self, **param):
        log = self.log
        with LogRedirect(self.log):
            log.working("Creating model...")
            if "embedding" in param:
                self.model_params['embedding'] = param['embedding']
                if not self.__create_embedding_model(**param["embedding"]): return
            if "umap" in param:
                self.model_params['umap'] = param['umap']
                if not self.__create_reduction_model(**param["umap"]): return
            if "hdbscan" in param:
                self.model_params['hdbscan'] = param['hdbscan']
                if not self.__create_clustering_model(**param["hdbscan"]): return
            if "projection" in param:
                self.model_params['projection'] = param['projection']
                if not self.__create_projection_model(**param["projection"]): return
        log.finished("Model created")
    
    @classmethod
    def load(cls, log:Logger, path:str):
        try:
            with open(path, 'rb') as file:
                model = joblib.load(file)
            model.log = log
            # when saving model, it is always in busy state so have to reset it now
            model.busy = False
            if model.embedding_model == True:
                model.__create_embedding_model(**model.model_params['embedding'])
            log.finished("Model loaded")
            return model
        except Exception:
            traceback.print_exc()
            log.error("Model loading failed")
            return None
    
    @LockModel
    def save(self, filename:str):
        # Do not save log as it is session dependent
        log = self.log
        self.log = None
        # Do not save embedding model as it is big and off-the-shelf
        embed_model = self.embedding_model
        self.embedding_model = (self.embedding_model is not None)
        self.group_tree = (self.group_tree is not None)

        try:
            with open(filename, 'wb') as file:
                joblib.dump(self, file)
            # restore vars
            self.log = log
            self.embedding_model = embed_model
            return True
        except Exception:
            traceback.print_exc()
            self.log = log
            return False

    def save_for_download(self, filename:str = "./saves/model.model", **kwargs):
        log = self.log
        if self.save(filename):
            log.finished("Model ready to be downloaded", {"path": filename})
        else:
            log.error("Model saving failed")

    @LockModel
    def load_from_csv(self, **param):
        log = self.log
        if not "text" in param:
            log.invalid()
            return
        try:
            usecols = [param['text']]
            names=['text']
            for col in ['ts', 'ref', 'target', 'distance', 'predict', 'set']:
                if col in param and not str(param[col]).startswith('fixed:'):
                    usecols.append(param[col])
                    names.append(col)
            
            filename= param["filename"] if "filename" in param else "uploaded.csv"
            df = pd.read_csv(filename, usecols=usecols, names=names, encoding='utf-8', header=None)
            
            for col in ['ts', 'ref', 'target', 'distance', 'predict', 'set']:
                if col in param and str(param[col]).startswith('fixed:'):
                    val = param[col].replace('fixed:', '')
                    df[col] = val
            
            if "ts" in df: df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
            df = df[df["text"].notna()]

            if 'set' in df.columns:
                self.devset = self.devset.append(df[df.set=='dev'].drop(columns=['set']), ignore_index=True)
                self.trainset = self.trainset.append(df[df.set!='dev'].drop(columns=['set']), ignore_index=True)
            else:
                self.trainset = self.trainset.append(df, ignore_index=True)
            log.finished("Dataset loaded")
        except Exception:
            traceback.print_exc()
            log.error("Dataset load error")
    
    @LockModel
    def load_from_json(self, **param):
        log = self.log
        if not 'text' in param:
            log.invalid()
            return
        try:
            filename= param["filename"] if "filename" in param else "uploaded.json"
            df = pd.read_json(filename, orient='index')
            columns = {}
            keepcolumn = []
            for col in ['text', 'ts', 'ref', 'target', 'predict', 'set']:
                if col in param and not str(param[col]).startswith('fixed:'):
                    columns[param[col]] = col
                    keepcolumn.append(param[col])
                df = df[keepcolumn].rename(columns = columns)
            for col in ['ts', 'ref', 'target', 'predict', 'set']:
                if col in param and str(param[col]).startswith('fixed:'):
                    val = param[col].replace('fixed:', '')
                    df[col] = val
            
            if "ts" in df: df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
            df = df[df["text"].notna()]

            if not 'text' in df.columns:
                log.error("Dataset has not text column")
                return
            if 'set' in df.columns:
                self.devset = self.devset.append(df[df.set=='dev'].drop(columns=['set']), ignore_index=True)
                self.trainset = self.trainset.append(df[df.set!='dev'].drop(columns=['set']), ignore_index=True)
            else:
                self.trainset = self.trainset.append(df, ignore_index=True)
            
            log.finished("Dataset loaded")
        except Exception:
            traceback.print_exc()
            log.error("Dataset load error")

    @LockModel
    def load_from_sqlite(self, **param):
        log = self.log
        if not 'text' in param:
            log.invalid()
            return
        filename = param["filename"] if "filename" in param else "uploaded.sqlite3"
        table = param["table"] if "table" in param else "dataset"
        
        try:
            con = sqlite3.connect(filename, detect_types=sqlite3.PARSE_DECLTYPES)
            cur = con.cursor()
            cur.execute("SELECT COUNT(*) FROM " + table)
            count = cur.fetchone()[0]
            log.working("Loading %d records" % count)
            sql = 'SELECT %s AS text' % (param['text'])
            for col in ['embedding', 'umap', 'x', 'y', 'z', 'ts', 'ref', 'target', 'predict']:
                if col in param and not str(param[col]).startswith('fixed:'):
                    sql += ', %s AS %s' % (param[col], col)
            sql += " FROM %s" % (table)
            df = pd.read_sql(sql, con)
            for col in ['ts', 'ref', 'target', 'predict', 'set']:
                if col in param and str(param[col]).startswith('fixed:'):
                    val = param[col].replace('fixed:', '')
                    df[col] = val
            if "ts" in df: df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
            df = df[df["text"].notna()]
            if 'set' in df.columns:
                self.devset = self.devset.append(df[df.set=='dev'].drop(columns=['set']), ignore_index=True)
                self.trainset = self.trainset.append(df[df.set!='dev'].drop(columns=['set']), ignore_index=True)
            else:
                self.trainset = self.trainset.append(df, ignore_index=True)
            log.finished("Dataset loaded")
        except Exception:
            traceback.print_exc()
            log.error("Dataset load error")

    def load_dataset(self, **param):
        if 'filename' not in param or 'type' not in param:
            self.log.invalid('Specify at least filename or file type')
        if 'type' in param:
            type = param['type']
            del param['type']
        else:
            import os
            _, type = os.path.splitext(param['filename'])
        if type == 'csv':
            self.load_from_csv(**param)
        elif type == 'json':
            self.load_from_json(**param)
        elif type in ['sqlite', 'sqlite3']:
            self.load_from_sqlite(**param)
        dataset = self.devset if "set" in param and param['set'] == 'dev' else self.trainset
        dataset.drop_duplicates(subset=['text'], inplace=True)
        dataset.dropna(subset=['text'], inplace=True)
        dataset.reset_index(drop=True, inplace=True)
        
    @LockModel
    def save_to_sqlite(self, filename:str=None, **kwargs):
        log = self.log
        if self.trainset.shape[0] == 0 and self.devset.shape[0] == 0:
            log.error("Nothing to save")
            return

        if filename is None:
            filename = 'saves/%s.sqlite3' % self.name

        try:
            con = sqlite3.connect(filename, detect_types=sqlite3.PARSE_DECLTYPES)
            dtypes = {
                "embedding": "array",
                "umap": "array"
            }
            self.trainset.to_sql("train", con, dtype=dtypes, if_exists='replace')
            self.devset.to_sql("test", con, dtype=dtypes, if_exists='replace')
            con.close()
        except Exception:
            traceback.print_exc()
            log.error("Dataset save error")

    @LockModel
    def save_to_csv(self, filename:str=None, **kwargs):
        log = self.log
        if self.trainset.shape[0] == 0:
            log.error("Nothing to save")
            return
        if filename is None:
            filename = 'saves/%s.sqlite3' % self.name
        
        log.working("Preparing dataset file, a moment...")
        try:
            with open(filename, "w", encoding='utf8', newline='') as f:
                self.trainset.to_csv(f, columns=["text", "ts", "ref", "target", "predict"], index=False)
            log.finished("Dataset ready to be downloaded", {"path": filename})
        except Exception:
            traceback.print_exc()
            log.error("Dataset save error")

    # DO NOT use log.finish as this is inner private function with no busy flag rights!
    def __create_embeddings(self, **param):
        log = self.log
        if self.embedding_model is None:
            log.invalid("Create embedding model first")
            return False
            
        dataset = self.devset if (param and "set" in param and param["set"] == 'dev') else self.trainset
        to_embed = dataset[dataset.embedding.isna()]
        if to_embed.shape[0] == 0:
            log.working("Nothing to do")
            return True
        log.working("Creating embeddings...")
        try:
            with LogRedirect(log):
                embs = self.embedding_model.encode(to_embed["text"].to_list(), show_progress_bar = True)
            for i in range(to_embed.shape[0]):
                dataset.at[to_embed.index[i], "embedding"] = embs[i]

            log.working("Finished embeddings")
            return True
        except Exception:
            traceback.print_exc()
            log.error("Embedding failed")
            return False
        
    # DO NOT use log.finish as this is inner private function with no busy flag rights!
    def __reduce_dimension(self, **param):
        log = self.log
        if self.reduction_model is None:
            log.invalid("Create reduction model first")
            return False
        istest = (param and "set" in param and param["set"] == 'dev')
        dataset = self.devset if istest else self.trainset
        to_reduce = dataset # you must do against the whole set!

        if to_reduce.shape[0] == 0:
            log.working("Nothing to do")
            return True
        
        if dataset["embedding"].isna().sum() > 0:
            log.invalid("Some rows not yet has embedding created, create embeddings first")
            return False

        log.working("Reducing dimension...")
        try:
            with LogRedirect(log):
                embeddings = to_reduce["embedding"].to_list()
                if not istest: 
                    if to_reduce["target"].notna().sum() > 0:
                        self.reduction_model.fit(embeddings, y=to_reduce["target"].fillna(-1))
                    else:
                        self.reduction_model.fit(embeddings)
                umaps = np.nan_to_num(self.reduction_model.transform(embeddings))
                for i in range(to_reduce.shape[0]):
                    dataset.at[to_reduce.index[i], "umap"] = umaps[i]
            log.working("Finished reduction")
            return True
        except Exception:
            traceback.print_exc()
            log.error("Reduction failed")
            return False

    # DO NOT use log.finish as this is inner private function with no busy flag rights!
    def __do_clustering(self, **param):
        log = self.log
        if self.clustering_model is None:
            log.invalid("Create clustering model first")
            return False
        istest = (param and "set" in param and param["set"] == 'dev')
        dataset = self.devset if istest else self.trainset
        to_cluster = dataset  # you must cluster the whole set!
        if to_cluster.shape[0] == 0:
            log.working("Nothing to do")
            return True
        
        if dataset["umap"].isna().sum() > 0:
            log.invalid("Some rows not yet has dimension reduced, do dimension reduction first")
            return False

        log.working("Clustering...")
        try:
            with LogRedirect(log):
                if not istest: 
                    self.clustering_model.fit(to_cluster["umap"].to_list())
                    clusters = self.clustering_model.labels_
                else:
                    import hdbscan
                    clusters, _ = hdbscan.approximate_predict(self.clustering_model, to_cluster['umap'].to_list())
                dataset.at[dataset.predict.isna(), "predict"] = clusters
            log.working("Finished clustering")
            return True
        except Exception:
            traceback.print_exc()
            log.error("Clustering failed")
            return False

    def __soft_clustering(self, **param):
        log = self.log
        if self.group_summaries is None:
            log.invalid("Fully train the model first")
            return False
        log.working("Clustering...")
        try:
            dataset = self.devset
            dists = euclidean_distances(X=dataset['umap'].to_list(), Y=np.array(self.group_summaries["mean"].to_list()))
            ranks = dists.argsort(axis=1)[:, 0]
            dists.sort(axis=1)
            dists = dists[:, 0]
            dataset['predict'] = ranks
            dataset['distance'] = dists
            log.working("Finished clustering")
            return True
        except Exception:
            traceback.print_exc()
            log.error("Clustering failed")
            return False

    # DO NOT use log.finish as this is inner private function with no busy flag rights!
    def __generate_projections(self, **param):
        log = self.log
        # if self.projection_model is None:
        #     self.__create_projection_model()
        
        # 15/11 Always generate a new model in case of model change
        self.__create_projection_model()
        istest = (param and "set" in param and param["set"] == 'dev')
        dataset = self.devset if istest else self.trainset
        to_project = dataset[dataset.x.isna()]
        if to_project.shape[0] == 0:
            log.working("Nothing to do")
            return True
        
        log.working("Generating 3D projections...")
        try:
            with LogRedirect(log):
                if not istest: self.projection_model.fit(dataset.umap.tolist(), y=dataset.predict.tolist())
                xyzs = self.projection_model.transform(dataset.umap.tolist())
                for i in range(to_project.shape[0]):
                    # dataset.at[to_project.index[i], ['x', 'y', 'z']] = xyzs[i]
                    dataset.at[to_project.index[i], ['x', 'y']] = xyzs[i]
            log.working("Finished projections")
            return True
        except Exception:
            traceback.print_exc()
            log.error("3D projections generation failed")
            return False
    
    # DO NOT use log.finish as this is inner private function with no busy flag rights!
    def __summarize_result(self, **kwargs):
        log = self.log

        try:
            dataset = self.trainset
            isdev = False

            if dataset["predict"].isna().sum() > 0:
                log.invalid("Some rows has not yet predicted, do prediction first")
                return False
                
            nogroup = dataset[dataset.predict == -1].predict.count()
            hasgroup = dataset.shape[0] - nogroup
            
            group_counts = dataset[dataset.predict != -1].groupby("predict")["text"].count()

            if dataset.x.isna().sum() > 0:
                self.__generate_projections(set="dev" if isdev else "train")

            log.working("Gathering infos...")

            group_mean = []
            group_x = []
            group_y = []
            # group_z = []
            group_distances = []
            group_rel_size = []
            group_repr_text = []
            for i in range(group_counts.shape[0]):
                data = dataset[dataset.predict == i]
                mean = data.umap.mean()
                distances = euclidean_distances(data.umap.to_list(), Y=[mean])
                x = data.x.mean()
                y = data.y.mean()
                # z = data.z.mean()
                repr_id = np.argmin(distances)
                distance = np.max(distances)-np.min(distances)

                group_mean.append(mean)
                group_distances.append(distance)
                group_x.append(x)
                group_y.append(y)
                # group_z.append(z)
                group_rel_size.append(distance)
                group_repr_text.append(data.iloc[repr_id].text)

            mean = np.mean(np.array(group_mean), axis=0)
            max_distance = float(np.max(euclidean_distances(np.array(group_mean), Y=[mean]))) * 2
            self.group_stats = {
                "num_group": int(group_counts.shape[0]),
                "no_group": int(nogroup),
                "has_group": int(hasgroup),
                "max_distance": max_distance,
            }
            self.group_summaries = pd.DataFrame(data={
                "group": np.arange(group_counts.shape[0]),
                "mean": group_mean,
                "x": group_x,
                "y": group_y,
                # "z": group_z,
                "count": group_counts,
                "distance": group_distances,
                "rel_size": group_rel_size / np.min(group_rel_size),
                "repr_text": group_repr_text,
            })

            log.working("Creating tree and graphs...")
            
            return self.__create_group_tree()
            
        except Exception:
            traceback.print_exc()
            log.error("Summary generation failed")
            return False

    # DO NOT use log.finish as this is inner private function with no busy flag rights!
    def __create_group_tree(self):
        group_summary = self.group_summaries
        group_counts = group_summary['count']
        try:
            if group_summary is None:
                return False
            try:
                cache = self.clustering_model.condensed_tree_
            except Exception:
                self.create_clustering_model()
                self.do_clustering()
                cache = self.clustering_model.condensed_tree_
            treedf = cache.to_pandas()
            # sel contains one text sample per group
            sel = treedf[treedf.child_size==1].groupby("parent").first()
            # s2 contains all nodes and group-leaves
            s2 = treedf[treedf.child_size > 1].copy()

            #
            # put real group number into pd frame
            #
            def applygroup(x):
                s2.loc[s2.child==int(x.name), "group"] = self.trainset.iloc[int(x.child)]["predict"]
            sel.apply(applygroup, axis=1)

            #
            # remove -1 group or repeated groups
            # Note: repeated groups are trimmed branches
            #
            def nullgroup(x):
                s2.loc[s2.child==x, "group"] = np.nan
            s2.loc[s2.group==-1, "group"] = np.nan
            for i in range(group_counts.shape[0]):
                s3 = s2[s2.group==i]["child"].sort_values()
                if s3.count() > 1:
                    s3.iloc[1:].apply(nullgroup)

            #
            # top bottom to build the tree
            # Note: s2 should be a table sorted top-bottom by default
            #
            cache = {}  # cache for a recusive tree
            condensed = []  # list for a condensed tree
            def topbottom(x):
                #
                # find parent
                # 
                parentid = int(x.parent)
                if parentid not in cache:
                    node = {"id": -parentid, "children": []}
                    cache[parentid] = node
                    self.group_tree = node
                    condensed.append({"id": -parentid, "parent": None})
                parent = cache[parentid]

                # 
                # add to parent as child
                #
                node = {
                    # "id": int(x.child),
                    "count": int(x.child_size),
                    # "lambda": float(x.lambda_val)
                }
                cnode = {
                    # "id": int(x.child),
                    "count": int(x.child_size),
                    'parent': -int(x.parent),
                }
                # add node to cache
                cache[int(x.child)] = node
                if 'children' not in parent:  # this is a trimmed node, skip it
                    return
                if not np.isnan(x.group):
                    group = int(x.group)
                    node["id"] = group
                    node["label"] = "[%d] %s" % (group, group_summary.loc[group, "repr_text"])
                    cnode['id'] = group
                    cnode['label'] = node["label"]
                else:
                    lamda = float(x.lambda_val)
                    node["id"] = -int(x.child)
                    node["label"] = "[%d] %f" % (int(x.child), lamda)
                    node["children"] = []
                    cnode['id'] = node["id"]
                    cnode['label'] = node["label"]
                # if "children" in parent:
                parent["children"].append(node)
                condensed.append(cnode)

            s2.apply(topbottom, axis=1)

            del cache
            return True
        except Exception:
            traceback.print_exc()
            self.group_tree = None
            self.log.error("Tree generation failed")
            return False

    @LockModel
    def train(self, **param):
        param['set'] = 'train'
        self.log.working("Begin model training...")
        cols_to_reset = {}
        if "reset_embedding" in param and param["reset_embedding"]:
            cols_to_reset["embedding"] = True
        if "reset_reduction" in param and param["reset_reduction"]:
            cols_to_reset["umap"] = True
        if "reset_clustering" in param and param["reset_clustering"]:
            cols_to_reset["predict"] = True
        if "reset_summarize" in param and param["reset_summarize"]:
            cols_to_reset["x"] = True
            cols_to_reset["y"] = True
            #cols_to_reset["z"] = True
        if len(cols_to_reset):
            cols_to_reset['set'] = 'train'
            if not self.__reset_columns(**cols_to_reset): return
        if "reset_summarize" in param and param["reset_summarize"]:
            self.group_summaries = None
            self.group_stats = None
            self.group_tree = None

        if not "embedding" in param or param["embedding"]:
            if not self.__create_embeddings(**param): return
        if not "reduction" in param or param["reduction"]:
            if not self.__reduce_dimension(**param): return
        if not "clustering" in param or param["clustering"]:
            if not self.__do_clustering(**param): return
        if not "summarize" in param or param["summarize"]:
            if not self.__summarize_result(**param): return
        self.log.finished("Finished model training")

    @LockModel
    def test(self, **param):
        param['set'] = 'dev'
        self.log.working("Begin testing with devset...")
        cols_to_reset = {}
        if "reset_embedding" in param and param["reset_embedding"]:
            cols_to_reset["embedding"] = True
        if "reset_reduction" in param and param["reset_reduction"]:
            cols_to_reset["umap"] = True
        if "reset_clustering" in param and param["reset_clustering"]:
            cols_to_reset["predict"] = True
        if "reset_summarize" in param and param["reset_summarize"]:
            cols_to_reset["x"] = True
            cols_to_reset["y"] = True
        if len(cols_to_reset):
            cols_to_reset['set'] = 'dev'
            if not self.__reset_columns(**cols_to_reset): return
        
        if not "embedding" in param or param["embedding"]:
            if not self.__create_embeddings(**param): return
        if not "reduction" in param or param["reduction"]:
            if not self.__reduce_dimension(**param): return
        if not "clustering" in param or param["clustering"]:
            if not self.__soft_clustering(**param): return
        if not "summarize" in param or param["summarize"]:
            if not self.__generate_projections(**param): return
        self.log.finished("Testing result ready")

    @LockModel
    def adhoc_predict(self, texts:list, **kwargs):
        log = self.log
        if self.embedding_model is None:
            log.invalid("Create embedding model first")
            return
        if self.reduction_model is None:
            log.invalid("Create reduction model first")
            return
        if self.trainset.predict.isna().sum() > 0:
            log.invalid("Do training first")
            return
        if self.group_summaries is None:
            log.invalid("Generate group summaries first")
            return
        try:
            log.working("Creating embeddings...")
            embeddings = self.embedding_model.encode(texts)
            log.working("Dimension reduction...")
            umaps = self.reduction_model.transform(embeddings)
            from pprint import pprint
            pprint(umaps)
            xyzs = self.projection_model.transform(umaps)
            xs = xyzs[:, 0]
            ys = xyzs[:, 1]
            #zs = xyzs[:, 2]
            log.working("Clustering...")
            #
            # 2 aproaches:
            #   1 is use approximate_predict from hdbscan
            #   1 is predict from distance info gathered from generate_group_stats
            #   (which should be the same as approximate_predict)
            #
            # # use approximate_predict
            # # pros: simple!
            # # 
            # from hdbscan import approximate_predict
            # predicts, _ = approximate_predict(self.clustering_model, umaps)

            #
            # predict from distance
            # pros: have soft clusters!
            #
            dists = euclidean_distances(X=umaps, Y=np.array(self.group_summaries["mean"].to_list()))
            ranks = dists.argsort(axis=1)[:, :3]
            dists.sort(axis=1)
            dists = dists[:, :3]
            predicts = [ranks[i][0] if dists[i][0] <= self.group_summaries.loc[ranks[i][0], "distance"] else -1 for i in range(len(texts))]

            adhocset = pd.DataFrame({'text': texts, 'x': xs, 'y': ys, 
                #'z': zs, 
                'predict': predicts})
            adhocset["embedding"] = np.nan
            adhocset['umap'] = np.nan
            adhocset["embedding"] = adhocset["embedding"].astype(object)
            adhocset["umap"] = adhocset["umap"].astype(object)
            for i in range(len(texts)):
                adhocset.at[i, 'embedding'] = embeddings[i]
                adhocset.at[i, 'umap'] = umaps[i]
            self.adhocset = adhocset
            #
            # for predict_from_distance
            #     put in probabilites
            adhocset[['choice1', 'choice2', 'choice3']] = ranks
            adhocset[['distance', 'distance2', 'distance3']] = dists
            
            # # for approximate_predict:
            # #     calculate your own distances
            # #
            # def dist(x):
            #     dist = euclidean_distances([x.umap], Y=[self.group_summaries["groups"].loc[x.predict, 'mean']])
            #     adhocset.loc[x.name, 'distance'] = dist
            # adhocset[adhocset.predict != -1].apply(dist, axis=1)

            log.finished("Result ready")
        except Exception:
            traceback.print_exc()
            log.error("Something wrong")


    @LockModel
    def read_train_result(self, **kwargs):
        log = self.log
        if self.group_summaries is not None and self.group_tree is not None:
            log.finished(param={
                "stats": {
                    "num_group": int(self.group_stats["num_group"]),
                    "no_group": int(self.group_stats["no_group"]),
                    "has_group": int(self.group_stats["has_group"]),
                    "max_distance": float(self.group_stats["max_distance"]),
                },
                "groups": self.group_summaries[["group","x", "y", 
                    #"z", 
                    "count", "distance", "rel_size", "repr_text"]].to_dict(orient="records"),
                "tree": self.group_tree
            })
        else:
            log.invalid(None)

    @LockModel
    def read_adhoc_predict_result(self, **kwargs):
        log = self.log
        if self.adhocset is None:
            log.invalid(None)
        result = self.adhocset.copy()
        del result["embedding"]
        del result['umap']
        log.finished(param={
            "prediction": result.to_dict(orient='records')
        })

if __name__ == "__main__":
    pass
# %%

    def iceptLog(status=-1000, msg=None, param=None):
        if not LogRedirect.out_redirected:
            print("[%d] %s" %(status, msg))
            if param: print(param)

# %%
    model = Model()
    model.log.log = iceptLog
    model.load_from_sqlite({
        "filename": "dev.sqlite3"
    })
    #model.create_embedding_model({"model": "all-MiniLM-L6-v2"})
    # model.load_from_sqlite_table({
    #     "filename": "devsample.sqlite3",
    #     "set": "train",
    #     "table": "dataset",
    #     "predict": "group"
    # })
    #model.dataset.loc[model.dataset.predict==-1, "embedding"] = np.nan
    #model.create_embeddings()
# %%
    log = Logger()
    log.log = iceptLog
    model = Model.load(log, "saves/checkpoints/Trump.model")
# %%
    model = Model()
    model.log.log = iceptLog
    model.load_from_csv(**{
        'filename':'rsics_dataset/tagged_selections_by_sentence.csv',
        'text': 5,
    })
    #model.train_test_split()
    model.create_model(**{
        'embedding': {
            # "model": "paraphrase-multilingual-mpnet-base-v2",
            "model": "all-MiniLM-L6-v2",
        },
        'umap': {
            'target_metric': 'cosine',
            'n_components': 10,
            'transform_seed': 42,
            'n_neighbors': 5,
            'min_dist': 1e-3,
            'target_weight': 0.5,
        },
        'hdbscan': {    
            'min_cluster_size': 5,
            'metric': 'euclidean',
            'cluster_selection_method': 'eom',
        }
    })
    model.train()

# %%
