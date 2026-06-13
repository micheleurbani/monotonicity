

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evidence_theory import core
import pandas as pd
import seaborn as sb


import pandas as pd
import seaborn as sb
from evidence_theory import core


def generate_datasets(N, m):
    """
    Returns `N` pandas DataFrame each containing a simulated use-case of
    evidence theory.

    Paramters
    ---------
    N: (int)
        The number of datasets to be generated.
    m: (int)
        The number of elements in the universal set `X`.

    Returns
    -------
    data: (list)
        A list of `pandas.DataFrame`.
    """
    data = []
    for i in range(N):
        data.append(
            core.generate_dataset(
                core.powerset(m),
            )
        )
    return data


def experiment(N, m, entropy_measures):
    """
    Returns `N` samples of the entropy measures in `entropy_measures`.
    """
    ds = generate_datasets(N, m)
    results = []
    for d in ds:
        data = {}
        for measure in entropy_measures:
            data[measure.__name__] = measure(d)
        results.append(data)
    return pd.DataFrame(results)


def plot_correlation(data):
    """
    Returns a grid of scatter plots representing the measure-to-measure
    comparisons.

    Paramters
    ---------
    data : pandas.DataFrame
    The dataset containing a sample of entropy measures.
    """
    fig = sb.pairplot(data)
    fig.savefig("scatter.png")
    return fig


if __name__ == "__main__":
    pass
