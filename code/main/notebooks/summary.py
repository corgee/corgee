import ast
import pandas as pd


def parse_metrics(m):
    p = ast.literal_eval(m)
    res = []
    for r in p:
        res.append([r['alignment'], r['uniformity_x'], r['uniformity_y'], r['uniformity_xy'],
                    r['rec_t'][0][1], r['rec_t'][1][1], r['rec_t'][2][1]])
    df = pd.DataFrame(res)
    df.columns = ['align', 'unifx', 'unify', 'unifxy', 'r@5', 'r@10', 'r@50']
    return df


def get_res(fname):
    lines = open(fname).readlines()
    lines = [line for line in lines if 'metrics: [' in line]
    print(len(lines))
    for line in lines:
        metrics = parse_metrics(line.split('metrics: [')[1].strip()[:-1])
        print(metrics)


log_fname = 'logs0.txt'
get_res(log_fname)
