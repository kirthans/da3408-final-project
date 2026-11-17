"""Versioned aggregation, conformal intervals and batched move occlusion."""

import json

import mlflow.pyfunc
import pandas as pd
import xgboost as xgb

from elo.models.aggregate import GAME_META_COLUMNS, perspective_rows, side_features


class EloPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.booster = xgb.Booster(params={"nthread": 2})
        self.booster.load_model(context.artifacts["model"])
        with open(context.artifacts["conformal"]) as stream:
            self.conformal = json.load(stream)
        with open(context.artifacts["features"]) as stream:
            self.columns = json.load(stream)

    def predict(self, context, model_input, params=None):
        top_k = (params or {}).get("top_k", 5)
        moves = model_input.copy()
        variants, removed = [moves], {}
        for gid, game in moves.groupby("game_id", sort=False):
            for index, move in game.iterrows():
                key = f"{gid}#{move.ply}"
                variant = game.drop(index).copy()
                variant["game_id"] = key
                variants.append(variant)
                removed[key] = (gid, move)
        # Full games and every leave-one-move-out variant share a single prediction.
        batch = pd.concat(variants, ignore_index=True)
        features = perspective_rows(side_features(batch), batch[GAME_META_COLUMNS])
        predictions = self.booster.predict(
            xgb.DMatrix(features[self.columns], enable_categorical=True)
        )
        lookup = {
            (row.game_id, "w" if row.is_white else "b"): float(pred)
            for row, pred in zip(features.itertuples(), predictions)
        }
        influence = {}
        for key, (gid, move) in removed.items():
            delta = lookup[(gid, move.side)] - lookup[(key, move.side)]
            influence.setdefault((gid, move.side), []).append(
                dict(ply=int(move.ply), san=move.san, side=move.side, delta_elo=delta)
            )
        output = []
        for gid, game in moves.groupby("game_id", sort=False):
            tc = game.iloc[0].tc_class
            q = self.conformal["q"].get(tc, self.conformal["global_q"])
            for side in ("w", "b"):
                pred = lookup[(gid, side)]
                items = sorted(influence.get((gid, side), []), key=lambda x: -abs(x["delta_elo"]))[
                    :top_k
                ]
                items = [{**item, "delta_elo": round(item["delta_elo"])} for item in items]
                output.append(
                    dict(
                        game_id=gid,
                        side=side,
                        elo=round(pred),
                        lo=round(pred - q),
                        hi=round(pred + q),
                        tc_class=tc,
                        influence=json.dumps(items),
                    )
                )
        return pd.DataFrame(output)
