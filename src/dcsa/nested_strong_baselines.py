"""Nested DOI-held-out tuning for strong global priors plus support Ridge."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from src.dcsa.baseline import _features, _load_points, _split_dois
from src.dcsa.data_source import PROJECT_ROOT, input_paths, sha256_file
from src.dcsa.masking_validation import select_support

GRIDS = {
    "extra_trees": [(120, 2, .8), (160, 4, 1.0)],
    "random_forest": [(120, 2, .8), (160, 4, 1.0)],
    "xgboost": [(160, 5, .04), (200, 6, .04)],
    "catboost": [(160, 6, .04), (200, 7, .04)],
}

def make_model(name, p, seed):
    if name == "extra_trees": return ExtraTreesRegressor(n_estimators=p[0], min_samples_leaf=p[1], max_features=p[2], random_state=seed, n_jobs=4)
    if name == "random_forest": return RandomForestRegressor(n_estimators=p[0], min_samples_leaf=p[1], max_features=p[2], random_state=seed, n_jobs=4)
    if name == "xgboost": return XGBRegressor(n_estimators=p[0], max_depth=p[1], learning_rate=p[2], subsample=.85, colsample_bytree=.85, objective="reg:squarederror", n_jobs=4, random_state=seed)
    return CatBoostRegressor(iterations=p[0], depth=p[1], learning_rate=p[2], loss_function="RMSE", verbose=False, random_seed=seed, thread_count=4)

def evaluate_prior(model, train, eval_points, valid, features, shot, seed):
    model.fit(_features(train, features), train.target.to_numpy(float))
    rows=[]
    eligible=valid.groupby("doi").filter(lambda g:g.material.nunique()>=shot+1)
    for doi, curves in eligible.groupby("doi", sort=True):
        sm,qm=select_support(curves, shot, "diverse_support", seed=seed)
        support=eval_points[eval_points.curve_row_id.isin(sm.curve_row_id)]
        query=eval_points[eval_points.curve_row_id.isin(qm.curve_row_id)]
        sb=model.predict(_features(support,features)); qb=model.predict(_features(query,features))
        st=support.temperature.to_numpy(float)/1000; qt=query.temperature.to_numpy(float)/1000
        ds=np.column_stack([np.ones(len(st)),st,st**2]); dq=np.column_stack([np.ones(len(qt)),qt,qt**2])
        pred=qb+Ridge(alpha=1.0).fit(ds,support.target.to_numpy(float)-sb).predict(dq)
        rows.append({"doi":doi,"mae":float(mean_absolute_error(query.target,pred))})
    return rows

def run(root: Path=PROJECT_ROOT, seeds=(11,22,33), shot=5):
    points,valid=_load_points(root); features=sorted(c for c in points if c.startswith("f_")); valid=valid.merge(points[["material",*features]].drop_duplicates("material"),on="material",how="inner"); all_dois=points.doi.unique(); tuned=[]; outer=[]
    for seed in seeds:
        outer_train,outer_test=_split_dois(all_dois,seed); inner_train,inner_val=_split_dois(np.array(sorted(outer_train)),seed+10000)
        tr=points[points.doi.isin(inner_train)]; va=points[points.doi.isin(inner_val)]; va_meta=valid[valid.doi.isin(inner_val)]
        chosen={}
        for name,grid in GRIDS.items():
            scores=[np.mean([r["mae"] for r in evaluate_prior(make_model(name,p,seed+10000),tr,points[points.doi.isin(inner_val)],va_meta,features,shot,seed+10000)]) for p in grid]
            idx=int(np.argmin(scores)); chosen[name]=grid[idx]; tuned.append({"seed":seed,"model":name,"params":repr(grid[idx]),"inner_mae":scores[idx]})
        test=points[points.doi.isin(outer_test)]; test_meta=valid[valid.doi.isin(outer_test)]
        for name,p in chosen.items():
            rows=evaluate_prior(make_model(name,p,seed),points[points.doi.isin(outer_train)],test,test_meta,features,shot,seed)
            for row in rows:
                outer.append({"seed":seed,"doi":row["doi"],"model":name+"+support_ridge","mae":row["mae"],"params":repr(p)})
        print(f"seed={seed} chosen={chosen}",flush=True)
    out=root/"results"; pd.DataFrame(tuned).to_csv(out/"nested_strong_baselines_tuning.csv",index=False); outer_frame=pd.DataFrame(outer); outer_frame.to_csv(out/"nested_strong_baselines_outer_doi.csv",index=False); outer_frame.groupby("model",as_index=False).agg(mae=("mae","mean"),median=("mae","median"),n_dois=("doi","nunique")).to_csv(out/"nested_strong_baselines_outer.csv",index=False)
    (out/"nested_strong_baselines_manifest.json").write_text(json.dumps({"seeds":list(seeds),"shot":shot,"inner_split":"DOI-held-out","outer_split":"DOI-held-out","grids":{k:[list(x) for x in v] for k,v in GRIDS.items()},"input_sha256":{n:sha256_file(p) for n,p in input_paths().items()}},indent=2),encoding="utf-8")
    return pd.DataFrame(outer)
if __name__=="__main__": print(run().to_string(index=False))
