from typing import Tuple

import os
import uuid
import json
from datetime import datetime

from multiprocessing import Pool
import tqdm

import numpy as np
import pandas as pd

from hyperminhash import HyperMinHash


def mult_range(start, end, step):
    cur = start

    while cur < end:
        yield cur
        cur *= step


def estimate_error(got, exp: int) -> float:
    if got == exp == 0:
        return 0.0
    if got != 0 and exp == 0:
        return 100.0

    if got > exp:
        delta = got - exp
    else:
        delta = exp - got

    return 100 * delta / exp


def run_experiment(p: int, q: int, r: int, max_card: int):
    card1, card2 = np.random.randint(0, max_card, size=2)
    card1, card2 = int(card1), int(card2)
    card_inter = np.random.randint(0, max(card1, card2))

    hll1 = HyperMinHash(p, q, r)
    hll2 = HyperMinHash(p, q, r)

    for i in range(card1):
        hll1.add(i)

    for i in range(card1 - card_inter, card1 - card_inter + card2):
        hll2.add(i)

    est_union = len(hll1.merge(hll2))
    est_inter = hll1.intersection(hll2)

    res = {
        "p": p, "q": q, "r": r, "max_card": max_card,
        "set1": card1, "set2": card2,
        "union": card1 + card2 - card_inter, "inter": card_inter,
        "est_union_val": int(est_union), "est_inter_val": int(est_inter)}
    res.update({
        "est_union_err": estimate_error(res["est_union_val"], res["union"]),
        "est_inter_err": estimate_error(res["est_inter_val"], res["inter"])})

    return res


class Job:
    def __init__(self, func, output_dir):
        self.func = func
        self.output_dir = output_dir

    def run(self, params: dict):
        uid = str(uuid.uuid1())
        start = datetime.now()

        try:
            res: dict = self.func(**params)
            err = ""
        except Exception as e:
            res = dict()
            err = str(e)

        end = datetime.now()
        res.update({"time_start": start.strftime("%Y-%m-%d %H:%M:%S"),
                    "time_end": end.strftime("%Y-%m-%d %H:%M:%S"),
                    "time_diff": (end - start).total_seconds(),
                    "uid": uid, "exception": err})

        with open(os.path.join(self.output_dir, uid + ".json"), "w") as fp:
            json.dump(res, fp)

        return res

    @staticmethod
    def collect_results(output_dir):

        data = []
        for f in os.listdir(output_dir):
            with open(os.path.join(output_dir, f), "r") as fp:
                data.append(json.load(fp))

        return pd.DataFrame(data)


def scan(ps: Tuple[int, int, int], qs: Tuple[int, int, int],
         rs: Tuple[int, int, int], max_cards: Tuple[int, int, int],
         num_iter: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    params = []
    for r in range(*rs):
        for q in range(*qs):
            for p in range(r, ps[1], ps[2]):
                for max_card in mult_range(*max_cards):
                    for _ in range(num_iter):
                        params.append({"p": p, "q": q, "r": r, "max_card": max_card})

    print("Found %d parameter permutations" % len(params))

    if len(params) == 0:
        return

    jb = Job(run_experiment, output_dir)

    with Pool() as p:
        gen = p.imap_unordered(jb.run, params, chunksize=1)
        res = [_ for _ in tqdm.tqdm(gen, total=len(params))]

        p.close()
        p.join()

    return res
