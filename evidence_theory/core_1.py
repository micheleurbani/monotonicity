
import numpy as np
import pandas as pd
import math
from copy import deepcopy


def namer(name):
    def wrapper(f):
        f.name = name
        return f
    return wrapper


def period(period):
    assert type(period) is int, f"'period' must be int and is {type(period)}."
    assert 1 <= period <= 4
    def wrapper(f):
        f.period = period
        return f
    return wrapper


def classification(classification):
    def wrapper(f):
        f.classification = classification
        return f
    return wrapper


def powerset(m):
    """
    Generates the power set of a frame of descerment with `m` elements.

    Parameters
    ----------
    m: (int)
        The number of elements in the frame of descernment `X`.

    Returns
    -------
    powerset: (numpy array)
        A boolean array with size :math:`2^m \\times m` that defines whether
        an element of `X` (on columns) belongs to the element (on rows) of the
        power set, and `m` is number of elements in `X`.
    """
    powerset = [np.zeros((m,))]
    for i in range(m):
        for j in range(len(powerset)):
            x = np.copy(powerset[j])
            x[i] = 1
            powerset.append(x)
    return np.stack(powerset)


def sample_mass(powerset):
    """
    Sample mass values for some elements of the powerset so that all the
    elements of the frame of descernment are covered by mass assignment.

    Parameters
    ----------
    powerset: (numpy array)
        A boolean array with size :math:`2^m \\times m` that defines whether
        an element of `X` (on columns) belongs to the element (on rows) of the
        power set, and `m` is number of elements in `X`.
    """
    # remember to exclude the empty set from mass assignment
    idxs = list(range(1, powerset.shape[0]))
    # sample a random element and add it to the list of mass assignements
    mass_assign = [idxs.pop(np.random.randint(len(idxs)))]
    x = powerset[mass_assign[0]].copy()
    if np.all(x):
        stop = True
    else:
        stop = False

    while not stop:
        mass_assign.append(idxs.pop(np.random.randint(len(idxs))))
        # compute the stopping condition
        x += powerset[mass_assign[-1]]
        if np.all(x):
            stop = True
    w = np.random.dirichlet(np.ones(len(mass_assign)))
    mass = np.zeros(powerset.shape[0])
    for i, idx in enumerate(mass_assign):
        mass[idx] = w[i]
    
    mass = np.round(mass, 2)
    return mass


def is_subset(a, b):
    """
    Returns True if `a` is a subset of `b`.
    """
    return np.array_equal(np.multiply(a, b), a)


def mass2belief(powerset, mass):
    """
    Compute the belief values from a mass values.
    """
    belief = np.zeros_like(mass)
    for i, e1 in enumerate(powerset):
        m = np.zeros_like(mass)
        for j, e2 in enumerate(powerset):
            if is_subset(e2, e1):  # Check if e2 is a subset of e1
                m[j] = mass[j]
        belief[i] = np.sum(m)
    return belief


def mass2plausibility(powerset, mass):
    """
    Compute the plausibility values from a mass values.
    """
    plausibility = np.zeros_like(mass)
    for i, e1 in enumerate(powerset):
        m = np.zeros_like(mass)
        for j, e2 in enumerate(powerset):
            if np.any(np.multiply(e1, e2)):
                m[j] = mass[j]
        plausibility[i] = np.sum(m)
    return plausibility


def cardinalities(powerset):
    """
    Compute the cardinality of each set of the powerset.
    """
    assert len(powerset.shape) == 2
    return np.sum(powerset, axis=1)


def commonality(powerset, mass):
    """
    Compute the commonality measure.
    """
    commonality = np.zeros_like(mass)
    for i, a in enumerate(powerset):
        m = np.zeros_like(mass)
        for j, b in enumerate(powerset):
            if is_subset(a, b):
                m[j] = mass[j]
        commonality[i] = np.sum(m)
    return commonality


def generate_dataset(powerset, mass=None):
    """
    Generate a list of dictionaries ready to be transformed to a Pandas
    DataFrame for visualization.
    A probability mass value is assigned to `known_mass_elements` elements of
    the power set and all the values of the mass sum to 1.

    Parameters
    ----------
    powerset: (numpy array)
        An boolean array with size :math:`m \\times 2^m` that define whether an
        element of `X` (on columns) belongs to the element (on rows) of the
        power set.

    mass: (numpy array)
        The mass values for each element of the powerset.

    Returns
    -------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing mass, belief, and plausibility values
        for the elements of the set.
    """
    if mass is None:
        mass = sample_mass(powerset)
    beliefe = mass2belief(powerset, mass)
    plausibility = mass2plausibility(powerset, mass)
    comm = commonality(powerset, mass)
    card = cardinalities(powerset)
    data = {
        "element": [i for i in powerset],
        "mass": mass,
        "belief": beliefe,
        "plausibility": plausibility,
        "commonality": comm,
        "card": card
    }
    df = pd.DataFrame(data)
    return df


@namer("Höhle (confusion)")
@period(1)
@classification('D, EB')
def hohle(data):
    """
    Computes the Hohle entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.

    """
    data = data[data.belief > 0] 
    
    data.loc[data.belief <= 1e-10, 'belief'] = 1e-10  
    return np.sum(data.mass * np.log2(1 / data.belief))


@namer("Yager (dissonance)")
@period(1)
@classification('D, EB')
def yager(data):
    """
    Computes the Yager entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    data = data[data.plausibility > 0]
    return np.sum(data.mass * np.log2(1 / data.plausibility))



@namer("Yager(non-specificity)")
@period(1)
@classification('D, EB')
def yager_non_specificity(data): 
    """Computes Yager's non-specificity.
        Parameters
    ----------
    powerset : numpy array
        The power set (powerset) of the frame of discernment.
    mass : numpy array
        The mass values associated with each element of the powerset.

    Returns
    -------
    non_specificity : float
        The value of Yager's non-specificity measure.    
    """

    non_specificity = 1 - np.sum(data.mass / data.card)
    return non_specificity




@namer("Smets")
@period(1)
@classification('D, EB')
def smets(data):
    """
    Computes the Smets entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    data = data[data.commonality > 0]
    entropy = 0.0
    entropy = np.sum(np.log2(1 / data.commonality))
    return entropy


@namer("Dubois & Prade (U-uncertainty)")
@period(1)
@classification('NS, EB')
def dubois_prade(data):
    """
    Computes the Dubois and Prade entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    data = data[data.mass > 0]
    return - np.sum(data.mass * np.log2(1 / data.card))


@namer("Hole (entropy of discenibleness) = Nguyen")
@period(1)
@classification('D, EB')
def nguyen(data):
    """
    Computes the Nguyen entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    data = data[data.mass > 0]
    return np.sum(data.mass * np.log2(1 / data.mass))


@namer("Dubois & Prade (entropy like index)")
@period(1) 
@classification('NS, EB') 
def dubois_prade_commonality(data):
    """
    Computes the Dubois & Prade entropy (entropy like index) of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    data = data[data.mass > 0]
    data = data[data.commonality > 0] 
    return - np.sum(data.mass * np.log(data.commonality)) 



@namer("Dubois & Prade (index of fuzziness)")
@period(1) 
@classification('NS, EB') 
def dubois_prade_fuzziness(data):
    """
    Computes the Dubois & Prade index of fuzziness of the powerset.
    d(m) = -ln(Σ m(A) * Q(A))

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    data = data[data.mass > 0] 
    data = data[data.commonality > 0]
    
    return -np.log(np.sum(data.mass * data.commonality))


@namer("Dubois & Prade (imprecision)")
@period(1)
@classification('NS, EB')
def dubois_prade_imprecision(data):
    """
    Computes the Dubois and Prade entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    data = data[data.mass > 0]
    return np.sum(data.mass * data.card)



@namer("Lamata & Moral lower")
@period(2)
@classification('TU, EB')
def lamata_moral(data):
    """
    Computes the Lamata and Moral entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    return yager(data) + dubois_prade(data)


@namer("Lamata & Moral upper")
@period(2)
@classification('TU, EB')
def lamata_moral_upper(data):
    """
    Computes the Lamata and Moral upper entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    term1 = 0.0
    for i in data.index:
        if data.loc[i].mass > 0:  
            max_plausibility = 0.0
            for j in range(len(data.loc[i].element)):
                if data.loc[i].element[j] == 1:  
                    for k in data.index:
                        if np.sum(data.loc[k].element) == 1 and data.loc[k].element[j] == 1:
                           max_plausibility = max(max_plausibility, data.loc[k].plausibility)
                           break 
            if max_plausibility > 0:  
                 term1 -= data.loc[i].mass * np.log2(max_plausibility)

    data = data[data.mass > 0] 
    term2 = 0.0
    if not data.empty: 
        sum_mass_card = np.sum(data.mass * data.card)
        if sum_mass_card > 0:
            term2 = np.log2(sum_mass_card)

    return term1 + term2


@namer("Klir & Ramer (discord)")
@period(2)
@classification('TU, EB')
def klir_and_ramer(data):
    """
    Computes the Klir and Ramer entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    data = data[data.mass > 0]
    kr = 0.0
    for i in data.index:
        x = 0.0
        for j in data.index:
            x += data.loc[j].mass * np.sum(data.loc[i].element * \
                data.loc[j].element) / data.loc[j].card
        kr -= data.loc[i].mass * np.log2(x)
    return kr


@namer("Klir Total Uncertainty")
@period(2) 
@classification('')  
def klir_total_uncertainty(data):
    """
    Computes Klir's Total Uncertainty measure.
    Total Uncertainty = Non-specificity (Dubois-Prade) + Discord (Klir & Ramer)

    Parameters
    ----------
    data: (pandas DataFrame)
        A pandas.DataFrame containing the mass, belief, and plausibility
        values of the elements of a powerset.

    Returns
    -------
    float
        The total uncertainty value.
    """
    return dubois_prade(data) + klir_and_ramer(data)



@namer("Inuiguchi = Harmanec & Klir")
@period(2)
@classification('TU, EB')
def harmanec_and_klir(data):
    m = len(data.iloc[0]["element"])
    data = data[data.mass > 0]
    AU = 0.0
    data = data.to_dict('records')

    while len(data) > 1:

        pset = powerset(len(data))
        UF = []
        for i in pset:
            if np.sum(i) >= 1:
                x = np.zeros((len(data[0]["element"]), ))
                belief = 0.0
                for j in np.nonzero(i)[0]:
                    x = np.maximum(x, data[j]["element"])
                    belief += data[j]["mass"]
                UF.append({
                    "element": x,
                    "belief": belief,
                    "card": np.sum(x)
                })

        idx = np.argmax([i["belief"] / i["card"] if i["card"] != 0 else 0 for i in UF]) 
        if isinstance(idx, np.ndarray) and len(idx) > 1:
            idx = idx[np.argmax([UF[i]["card"] for i in idx])]
        A = UF[idx]
        p_x = A["belief"] / np.sum(A["element"]) if np.sum(A["element"]) != 0 else 0

        AU_change = (A["belief"] * np.log2(p_x)) if p_x > 0 else 0 

        AU -= AU_change

        F = [{
                "element": np.maximum(
                    np.subtract(A_i["element"], A["element"]),
                    np.zeros_like(A_i["element"])
                )
            } for A_i in data
        ]

        F = [i for i in F if np.any(i["element"])]

        seen_elements = set()
        deduped_F = []
        for item in F:
            element_tuple = tuple(item["element"])
            if element_tuple not in seen_elements:
                seen_elements.add(element_tuple)
                deduped_F.append(item)
        F = deduped_F

        for i, S in enumerate(F):
            S["mass"] = np.sum([
                A_i["mass"] for A_i in data if not np.any(
                    np.subtract(
                        S["element"],
                        np.maximum(
                            np.subtract(A_i["element"], A["element"]),
                            np.zeros_like(A_i["element"])
                        )
                    ))])
            S["card"] = np.sum(S["element"])

        data = deepcopy(F) 
    if len(data) == 1:  
        if data[0]["card"] !=0: 
            AU_change = data[0]["mass"] * np.log2(data[0]["mass"] / data[0]["card"]) if data[0]["mass"] >0 and data[0]["card"] > 0 else 0
            AU -= AU_change
    return AU


@namer("Klir & Parviz")
@period(2)
@classification('TU, EB')
def klir_and_parviz(data):
    """
    Computes the Klir and Parviz entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    data = data[data.mass > 0]
    kp = 0.0
    for i in data.index:
        x = 0.0
        for j in data.index:
            x += data.loc[j].mass * np.sum(data.loc[i].element * \
                data.loc[j].element) / data.loc[i].card
        kp -= data.loc[i].mass * np.log2(x)
    return kp


@namer("Pal et al.")
@period(2)
@classification('TU, EB')
def pal_et_al(data):
    """
    Computes the Pal et al. entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    data = data[data.mass > 0]
    return np.sum(data.mass * np.log2(data.card / data.mass))


@namer("Maeda & Hichihashi")
@period(2)  
@classification('TU, EB')  
def maeda_hichihashi(data):
    """
    Computes the Maeda & Hichihashi entropy of the powerset.
    Maeda & Hichihashi = Harmanec & Klir + Dubois & Prade (U-uncertainty)

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    return harmanec_and_klir(data) + dubois_prade(data)


@namer("George & Pal")
@period(2)
@classification('TU')
def george_and_pal(data):
    """
    Computes the George and Pal entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    data = data[data.mass > 0]
    gp = 0.0
    for i in data.index:
        a = data.loc[i]
        int_sum = 0.0
        for j in data.index:
            b = data.loc[j]
            int_sum += b.mass * (1 - (np.sum(a.element * b.element) / np.sum((a.element + b.element) > 0)))
        gp += a.mass * int_sum
    return 1 - gp


@namer("Maluf")
@period(3)  
@classification('D, EB') 
def maluf(data):
    """
    Computes the Maluf entropy measure Hds(X) of the powerset.
    Hds(X) = - Σ Pl(y) * log₂(Bel(y))

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.

    Returns
    -------
    float
        The Maluf entropy value.
    """
    data = data[data.belief > 0]  

    return -np.sum(data.plausibility * np.log2(data.belief))


@namer("Klir (Shannon-like)")
@period(3)
@classification('D, EB')
def klir_shannon(data):
    """
    Computes the Klir (Shannon-like) entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.

    Returns
    -------
    float
        The Klir (Shannon-like) entropy value.
    """

    data_singletons = data[data.card == 1]  

    c = np.sum(data_singletons.belief + data_singletons.plausibility) 

    entropy = 0.0
    for _, row in data_singletons.iterrows():  
        bel = row.belief
        pl = row.plausibility

        if pl > 0: 
            entropy -= (pl * np.log2(pl)) / c
        if bel > 0:
            entropy -= (bel * np.log2(bel)) / c
            
    return entropy



@namer("Yager-Shapley") 
@period(3)
@classification('D, EB') 
def yager_shapley(data):
    """
    Computes the Yager-Shapley entropy of the powerset.
    
    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.  It is assumed that 'element'
        columns represent the subsets and 'mass' column represents the mass values.

    Returns
    -------
    float
        The Yager-Shapley entropy value.
    """
    n = len(data.iloc[0]['element']) # fod size
    entropy = 0.0

    for i in range(n):  
        inner_sum = 0.0
        for j, row in data.iterrows():  
            if row['element'][i] == 1:  
                inner_sum += row['mass'] / row['card']

        if inner_sum > 0:  
            entropy -= inner_sum * np.log(inner_sum)


    return entropy


@namer("Jousselme et al.")
@period(3)
@classification('TU, EB')
def jousselme_et_al(data):
    """
    Computes the Jousselme et al. entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    am = 0.0
    for i in data.index:
        if np.sum(data.loc[i].element) == 1:
            bet = 0.0
            for j in data.index:
                if is_subset(data.loc[i].element, data.loc[j].element):
                    bet += data.loc[j].mass / data.loc[j].card
            if bet != 0.0:
                am += bet * np.log2(bet)
    return - am


@namer("Yang & Han")
@period(4)
@classification('TU, IB')
def yang_and_han(data):
    """
    Computes the Yan and Hang entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    tu = 0.0
    for i in data.index:
        x = data.loc[i]
        if np.sum(x.element) == 1:
            tu += np.sqrt(
                ((x.belief + x.plausibility) / 2 - 1 / 2)**2 + \
                    1 / 3 * ((x.plausibility - x.belief) / 2 - 1 / 2)**2
            )
    return 1 - (1 / len(data.iloc[0].element)) * np.sqrt(3) * tu


@namer("Deng")
@period(4)
@classification('TU, EB')
def deng(data):
    """
    Computes the Deng entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    data = data[data.mass > 0]
    return nguyen(data) + np.sum(data.mass * np.log2(np.power(2, data.card) - 1))


@namer("Wang & Song")
@period(4)
@classification('TU, IB')
def wang_and_song(data):
    """
    Computes the Wang and Sung entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    ws = 0.0
    for i in data.index:
        a = data.loc[i]
        if np.sum(a.element) == 1 and a.belief + a.plausibility != 0:
            ws += - ((a.belief + a.plausibility) / 2) * \
                np.log2((a.belief + a.plausibility) / 2) + \
                    (a.plausibility - a.belief) / 2
    return ws


@namer("Zhou et al.")
@period(4)
@classification('TU, EB')
def zhou_et_al(data):
    """
    Computes the Zhou et al. entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    data = data[data.mass > 0]
    card_X = len(data.iloc[0].element)
    return - np.sum(np.multiply(
        data.mass,
        np.log2(
            data.mass / (2**data.card - 1)) * np.exp((data.card - 1) / card_X)))



@namer("Tang")
@period(4)  
@classification('TU, EB')  
def tang(data):
    """
    Computes the Tang entropy of the powerset.
    
    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.

    Returns
    -------
    float
        The Tang entropy value.
    """
    data = data[data.mass > 0]  
    card_X = len(data.iloc[0].element)  
    tang_entropy = 0.0
    
    for _, row in data.iterrows():
        mass = row['mass']
        card_A = row['card']

        if mass > 0 and card_A > 0 and 2**card_A - 1 > 0: 
            tang_entropy -= (card_A * mass / card_X) * np.log2(mass / (2**card_A - 1))

    return tang_entropy



@namer("Pan & Deng")
@period(4)
@classification('TU, EB')
def pan_and_deng(data):
    """
    Computes the Pan and Deng entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    data = data[data.mass > 0]
    return - np.sum(
        np.multiply(
            (data.belief + data.plausibility) / 2,
            np.log2(
                (data.belief + data.plausibility) / (2 * (2**data.card - 1))
            )
        )
    )


@namer("Jiroušek & Shenoy (Pl Pr entropy)")
@period(4)
@classification('TU, EB')
def jirousek_and_shenoy(data):
    """
    Computes the Jirousek and Shenoy entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    K = 0.0
    pl_pm = []
    for i in data.index:
        if np.sum(data.loc[i].element) == 1:
            K += data.loc[i].plausibility
            pl_pm.append(data.loc[i].plausibility)
    pl_pm = [i for i in pl_pm if i != 0]
    pl_pm = np.array(pl_pm) / K
    return - np.dot(pl_pm, np.log2(pl_pm)) + dubois_prade(data)



@namer("Jiroušek & Shenoy (q entropy)")
@period(4)  
@classification('TU, EB') 
def jirousek_shenoy_q_entropy(data):
    """
    Computes the Jiroušek and Shenoy (q entropy) of the powerset.
    
    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset. It also needs the 'commonality' precomputed.


    Returns
    -------
    float
        The Jiroušek & Shenoy (q entropy) value.
    """

    q_entropy = 0.0
    for _, row in data.iterrows():
        q_a = row['commonality']
        #print("q_a:", q_a)
        cardinality_a = row['card'] 
        if q_a > 0:  
            q_entropy += (-1)**cardinality_a * q_a * np.log2(q_a) 
    return q_entropy


@namer("Mambé")
@period(4)  
@classification('TU, EB')  
def mambe(data):
    """
    Computes the Mambé entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.

    Returns
    -------
    float
        The Mambé entropy value.
    """
    data = data[data.mass > 0]  
    card_X = len(data.iloc[0].element)  
    mambe_entropy = 0.0

    for _, row in data.iterrows():
        mass = row['mass']
        card_A = row['card']

        if mass > 0 and card_A > 0:  
            log_arg = (mass / (2**card_A - 1)) * np.exp((card_A - 1) / (2**card_X))
            mambe_entropy -= mass * np.log2(log_arg)

    return mambe_entropy



@namer("Cui et al.")
@period(4)
@classification('TU, EB')
def cui_et_al(data):
    """
    Computes the Cui et. al. entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    data = data[data.mass > 0]
    card_X = len(data.iloc[0].element)
    cui = 0.0
    for i in data.index:
        int_sum = 0.0
        for j in data.index:
            if not np.array_equal(data.loc[i].element, data.loc[j].element):
                int_sum += np.sum(data.loc[i].element * data.loc[j].element) \
                    / (2**card_X - 1)
        cui += data.loc[i].mass * np.log2(
            (data.loc[i].mass / (2**data.loc[i].card - 1)) * np.exp(int_sum)
        )
    return - cui


@namer("Li")  
@period(4)       
@classification('NS, EB')  
def li(data):
    """
    Computes the Li's measure of Imprecision.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass values of the elements of a powerset.

    Returns
    -------
    float
        The Li imprecision value.
    """
    li = 0.0
    data = data[data.mass > 0]
    for _, row in data.iterrows():
        mass = row['mass']
        card_a = row['card']
        if card_a > 0 :  
            li += (mass / (2**card_a - 1))**2
    return li



@namer("Pan (2nd)")
@period(4)  
@classification('TU, EB')  
def pan_2nd_entropy(data):
    """
    Computes the Pan 2nd entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, plausibility
        values, and the 'element' column of the elements of a powerset.

    Returns
    -------
    float
        The Pan 2nd entropy value.
    """

    n = len(data.columns) -1 # Number of elements in the frame of discernment (excluding the 'mass' column)
    singleton_plausibilities = data[data['card'] == 1]['plausibility'].values
    total_plausibility = np.sum(singleton_plausibilities)
      
    pt = singleton_plausibilities / total_plausibility

    pm = np.zeros(len(data))  
    for i, row in data.iterrows():
        subset_elements = row.element
        pm[i] = np.sum(pt[subset_elements.astype(bool)]) 

    entropy = 0.0
    for i, row in data.iterrows():
       if pm[i] > 0: 
            entropy += row['mass'] * np.log2(1 / pm[i])

    return entropy + dubois_prade(data)



@namer("Chen")
@period(4)  
@classification('TU, EB')  
def chen_entropy(data):
    """
    Computes the Chen entropy of the powerset.
    
    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass and cardinality values of the elements of a powerset.

    Returns
    -------
    float
        The Chen entropy value.
    """
    data = data[data.mass > 0]  
    card_U = len(data.iloc[0]['element'])
    chen_ent = 0.0

    for _, row in data.iterrows():
        mass = row['mass']
        card_ui = row['card']
        if mass > 0 and card_ui > 0 and 2**card_ui - 1 > 0: 
            chen_ent -= mass * np.log2((mass / (2**card_ui - 1)) * (card_ui / card_U)) 
            
    return chen_ent


@namer("Gao")
@period(4)
@classification('TU, EB')
def gao(data):
    """
    Computes the Gao entropy measure U_G of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass and cardinality
        values of the elements of a powerset.

    Returns
    -------
    float
        The Gao entropy value.
    """
    gao_entropy = 0.0

    for _, row in data.iterrows():
        mass_A = row['mass']
        card_A = row['card']

        if card_A == 1 and mass_A > 0:
            gao_entropy += np.sum(mass_A * np.log2(1 / mass_A))


        elif card_A > 1:
            term1 = (2**card_A - 1) * mass_A
            term2 = 1 - (mass_A / (2**card_A - 1))**(card_A - 1)
            term3 = 1 / (card_A-1)
            gao_entropy += term1 * term2 * term3

    return gao_entropy


@namer("Zhao")
@period(4)  
@classification('TU, EB')  
def zhao_entropy(data):
    """
    Computes the Zhao entropy (Hinter(m)) of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A pandas DataFrame containing the mass, belief, plausibility, and 'element' columns
        of the elements of a powerset.

    Returns
    -------
    float
        The Zhao entropy value.
    """

    zhao_ent = 0.0
    n = len(data.iloc[0]['element'])

    for _, row in data.iterrows():
        mass_x = row['mass']
        card_x = row['card']
        belief_x = row['belief']
        plausibility_x = row['plausibility']

        if card_x == 1:
            avg_plausibility_belief = (belief_x + plausibility_x) / 2
            if avg_plausibility_belief > 0:
                zhao_ent -= avg_plausibility_belief * np.log2(avg_plausibility_belief * np.exp(-(plausibility_x - belief_x)))

    
        if mass_x > 0 and card_x > 1:  # Second summation for non-singletons X, |X|>1
            zhao_ent -= mass_x * np.log2((mass_x / (2**card_x - 1)) * np.exp(-(plausibility_x - belief_x)))

    return zhao_ent



@namer("Yan & Deng")
@period(4)
@classification('TU, EB')
def yan_and_deng(data):
    """
    Computes the Yan and Deng entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    S = 0
    for i in data.index:
        if np.sum(data.loc[i].element) == 1 and data.loc[i].plausibility > 0:
            S += 1

    data = data[data.mass > 0]
    return - np.sum(
        data.mass * np.log2(
            np.multiply(
                (data.mass + data.belief) / \
                    (2 * (np.power(2, data.card) - 1)),
                np.exp((data.card - 1) / S)
            )
        )
    )



@namer("Qin et al.")
@period(4)
@classification('TU, EB')
def qin_et_al(data):
    """
    Computes the Quin et al. entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    data = data[data.mass > 0]
    return np.sum((data.card / len(data.iloc[0].element)) * data.mass * \
        np.log2(data.card)) + nguyen(data)



@namer("Li & Pan")
@period(4)
@classification('TU, EB')
def li_and_pan(data):
    """
    Computes the Li and Pan entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    data = data[data.mass > 0]
    return np.sum(
        np.multiply(
            data.mass,
            np.log2(data.card**len(data.iloc[0].element) / data.mass)
        )
    )


@namer("Li (Improved)")
@period(4)  
@classification('NS, EB') 
def li_improved(data):
    """
    Computes the Li Improved entropy (IQmi) of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A pandas DataFrame containing the mass and 'element' columns of the
        elements of a powerset.

    Returns
    -------
    float
        The Li Improved entropy value.
    """
    n = len(data.iloc[0]['element'])  # Cardinality of the frame of discernment |X|
    li_imp = 0.0

    for i, row_a in data.iterrows():
        mass_a = row_a['mass']
        card_a = row_a['card']
        element_a = row_a['element']

        if card_a > 0:  # Avoid division by zero for the empty set
            inner_sum = 0.0
            for j, row_b in data.iterrows():
                mass_b = row_b['mass']
                if i != j and mass_a > 0 and mass_b > 0:  # B ≠ A
                    element_b = row_b['element']
                    intersection_cardinality = np.sum(element_a * element_b)  # |A∩B|
                    inner_sum += intersection_cardinality / n

            li_imp += ((mass_a / (2**card_a - 1))**2) * np.exp(inner_sum)

    return li_imp



@namer("Wen")
@period(4)
@classification('TU, EB')
def wen_entropy(data):
    """
    Computes the Wen entropy (U_exp(m)) of the powerset.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with 'mass', 'card', and 'element' columns.

    Returns
    -------
    float
        The Wen entropy value.
    """
    sum_singletons = 0.0
    sum_non_singletons = 0.0

    # Calculate |Θ| (cardinality of the frame of discernment) and the denominator
    n = len(data.iloc[0]['element'])  
    denominator = np.exp(1) -(1/(n**2)) * np.exp(1/(n**2))


    # Calculate the core (C)
    core_elements = np.zeros_like(data.iloc[0]['element'])  
    for _, row in data[data['mass'] > 0].iterrows():
        core_elements = np.logical_or(core_elements, row['element']).astype(int)

    card_c = np.sum(core_elements)

    for _, row in data.iterrows():
        mass_a = row['mass']
        card_a = row['card']

        if card_a == 1:
            sum_singletons += mass_a * np.exp(mass_a)
        elif mass_a > 0 and card_a > 1:
            sum_non_singletons += (mass_a / (card_a * card_c)) * np.exp(mass_a / (card_a * card_c))

    wen_ent = (np.exp(1) -sum_singletons - sum_non_singletons) / denominator

    return wen_ent


@namer("Chen improved") 
@period(4)  
@classification('TU, EB') 
def chen_improved(data):
    """
    Computes the custom entropy measure EWd(m^-i) of the powerset.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with 'mass', 'card', and 'element' columns.

    Returns
    -------
    float
        The custom entropy value.

    """
    n = len(data[data['mass'] > 0])  
    card_X = len(data.iloc[0]['element'])  
    custom_ent = 0.0

    for i, row in data.iterrows():
        mass_a = row['mass']
        card_a = row['card']
        print("card_a:", card_a)
        

        if n > 1 and mass_a > 0:  
          negated_mass_a = (1- mass_a)/(n-1)
          print("negated_mass_a:", negated_mass_a)
          

          if negated_mass_a > 0 and card_a > 0:  
            custom_ent -= (card_a * negated_mass_a / card_X) * np.log2(negated_mass_a / (2**card_a - 1))

    return custom_ent



@namer("Deng & Wang")
@period(4)
@classification('TU, IB')
def deng_and_wang(data):
    """
    Computes the Deng and Wang entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    du = 0.0
    for i in data.index:
        if np.sum(data.loc[i].element) == 1:
            du += 1 - np.sqrt(data.loc[i].belief + data.loc[i].plausibility + \
                1 - 2 * np.sqrt(data.loc[i].plausibility))
    return du


@namer("Dezert & Tchamova (Extended BetP)")
@period(4)
@classification('TU, EB')
def dezert_tchamova_betp(data, epsilon=1e-6):
    """
    Computes the Dezert & Tchamova extended BetP entropy of the powerset.

    H(m) = - Σ_{θᵢ∈Θ} BetP(θᵢ) * log(BetP(θᵢ)) + U(m)

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with 'mass', 'card', 'element' and 'betp' columns.
    epsilon : float, optional
        Small positive value to handle cases where BetP might be 0. The default is 1e-6.


    Returns
    -------
    float
        The extended BetP entropy value.
    """

    return jousselme_et_al(data) + dubois_prade(data)


@namer("Zhang et al.")
@period(4)
@classification('TU, EB')
def zhang(data):
    """
    Computes the Zhang et al. entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the belief and plausibility
        values of the elements of a powerset.

    Returns
    -------
    float
        The Zhang et al. entropy value.
    """
    n = len(data.iloc[0]['element'])  # Size of FOD
    ZU = 0.0

    for i in range(n): # Iterate for each element of the powerset
        bel = data.loc[i+1].belief
        pl = data.loc[i+1].plausibility

        d = np.sqrt(((bel - 0)**2) + ((pl - 1)**2))
       
        ZU += (4/3) * (1/ (1 + d)**2 - 1/4)  

    return ZU



@namer("Li et al.")
@period(4)
@classification('TU, IB')
def li_et_al(data):
    """
    Computes the Li et al. entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    li = 0.0
    for i in data.index:
        if np.sum(data.loc[i].element) == 1: 
            
            d_E = np.sqrt(
                data.loc[i].belief**2 + (data.loc[i].plausibility - 1)**2
            )
            li += (2 / (1 + d_E)) - 1
    return li


@namer("Zhou & Deng")
@period(4)
@classification('TU, EB')
def fractal_based_entropy(data):
    """
    Computes the fractal based entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    data = data[data.card > 0]  
    m_F = np.zeros(len(data)) 
    for i in data.index:
        for j in data.index:
            if is_subset(data.loc[i].element, data.loc[j].element):
             
                m_F[data.index.get_loc(i)] += data.loc[j].mass / (2**data.loc[j].card - 1)

    
    data = data[m_F > 0] 

    
    m_F = m_F[m_F > 0]

    return - np.sum(m_F * np.log2(m_F))


@namer("Zhou belief entropy")
@period(4)
@classification('D, EB')
def zhou_belief_entropy(data):
    """
    Computes the Belief Entropy measure.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass values of elements in the powerset.
    """
     # Extract focal elements (sets with nonzero mass)
    focal_elements = data[data.mass > 0].copy()
    B = focal_elements["element"].values  # List of focal sets
    m = focal_elements["mass"].values  # Corresponding mass values
    card_A = focal_elements["card"].values  # Corrected cardinalities

    print("Focal elements (B):", len(B))
    # print("Mass values (m):", m)
    # print("Cardinalities (card_A):", card_A)

    # If only one focal element exists, use the simplified formula
    if len(B) == 1:
        A = B[0]
        belief_entropy_value = -m[0] * np.log2(m[0] / (2 ** card_A[0] - 1))
        return belief_entropy_value

    elif len(B) > 1:
        # Compute structural conflict (SC) value for each pair of focal elements
        SC_matrix = np.zeros((len(B), len(B)))
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    Ai, Aj = B[i], B[j]
                    intersection = np.logical_and(Ai, Aj).sum()
                    union = np.logical_or(Ai, Aj).sum()
                    SC_matrix[i, j] = 1 - (intersection / union) if union > 0 else 1
        
        # Compute SC weight for each focal element by summing its row
        SC_values = np.sum(SC_matrix, axis=1)

        # Compute entropy term using corrected cardinalities
        entropy_terms = np.zeros(len(B))
        for i in range(len(B)):
            if m[i] > 0:
                entropy_terms[i] = -m[i] * np.log2(m[i] / (2 ** card_A[i] - 1))

        # Compute belief entropy measure
        belief_entropy_value = (1 / (len(B) - 1)) * np.sum(SC_values * entropy_terms)

    return belief_entropy_value


@namer("Xue & Deng (U_XD)")
@period(4)
@classification('TU, EB')
def xue_deng_entropy(data):
    """
    Computes the Xue & Deng entropy U_XD(m) of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing mass, belief, plausibility, and commonality values.

    Returns
    -------
    float
        The Xue & Deng entropy value.
    """
    entropy = 0.0
    for _, row in data.iterrows():
        q_a = row['commonality']
        print("q_a:", q_a)
        card_a = row['card']

        if q_a > 0 and card_a > 0:
            entropy += (-1) ** card_a * q_a * np.log2(q_a / (2**card_a - 1))

    return entropy


@namer("Dutta & Shome")
@period(4)  
@classification('TU, EB')  
def dutta_and_shome(data):
    """
    Computes the Dutta & Shome belief entropy measure EP of the powerset.
    """
    data = data[data.mass > 0]

    card_X = len(data.iloc[0]['element'])
    core_elements = np.zeros_like(data.iloc[0]['element'])  
    for _, row in data[data['mass'] > 0].iterrows():
        core_elements = np.logical_or(core_elements, row['element']).astype(int)

    card_c = np.sum(core_elements)
    print("card_unioneF:", card_c)
   
    common_element = np.ones(card_X, dtype=bool)

    for _, row in data.iterrows():
        common_element = np.logical_and(common_element, row['element'])

    card_common_element = np.sum(common_element)

    dut = 0.0

    for _, row in data.iterrows():
        mass = row['mass']
        card_ui = row['card']

        if mass > 0 and card_ui > 0: 
            dut -= mass * np.log2((mass / (2**card_ui - 1)) * (card_ui / card_c)*(np.exp(1+card_common_element/card_c))/card_c)         
            
    return dut





@namer("New Cheng & Deng")
@period(4)  
@classification('TU, EB') 
def new_cheng_deng(data):
    """
    Computes the new entropy measure En(m) of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset. It is assumed that 'element'
        columns represent the subsets and 'mass' column represents the mass values.

    Returns
    -------
    float
        The new entropy value En(m).
    """
    
    entropy1 = 0.0
    subsets = {}
    for _, row in data.iterrows():
        mass = row['mass']
        card = row['card']
        arg = 0.0
        if mass > 0:
            for i in range(int(card)):
                subsets[i] = math.comb(int(card), i+1)
                arg += (2**(i+1) -1) * subsets[i]
                            
            entropy1 += mass * np.log2(arg)


    entropy2 = 0.0
    for _, row in data.iterrows():
        mass = row['mass']
        if mass > 0 : 
            entropy2 -= mass * np.log2(mass)

    return entropy1 + entropy2


@namer("New Cui & Deng")
@period(4)  
@classification('TU, EB')  
def cui_and_deng(data):
    """
    Computes the new measure HPl(m) of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    data_singletons = data[data.card == 1] # Filter for singletons
    data_singletons = data_singletons[data_singletons.plausibility > 0] # plausibility > 0
    total_plausibility = data_singletons.plausibility.sum()

    pla = 0.0
    for i in data_singletons.index:
        plausibility = data_singletons.loc[i].plausibility
        pla -= plausibility * np.log2(plausibility / total_plausibility)

    return pla


@namer("Kavya et al.")
@period(4)  
@classification('TU, EB') 
def kavya_et_al(data):

    """
    Computes the Kavya et al. entropy measure of the powerset.
    
    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.

    Returns
    -------
    float:
        The calculated Kavya et al. entropy.
    """

    data = data[(data.card == 1)]

    return np.sum((-data.plausibility * np.log2(data.plausibility))/(np.exp(data.plausibility - data.belief)) + (data.plausibility - data.belief))


@namer("Zhou & Deng (U_SU')")
@period(4)
@classification('TU, EB')
def zhou_deng_discord(data):
    """
    Computes the discord part of the Zhou & Deng entropy.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing belief, plausibility, and mass values.

    """
    data = data[data.card == 1]
    
    normalization_term = np.sum(data["plausibility"] + data["belief"])

    if normalization_term == 0:
        return 0.0
 
    entropy = 0.0
    for _, row in data.iterrows():
        pl_bel_sum = row["plausibility"] + row["belief"]
        if pl_bel_sum > 0:
            prob = pl_bel_sum / normalization_term
            entropy -= prob * np.log2(prob)

    return entropy 

@namer("Zhou & Deng (U_ZD)")
@period(4)
@classification('TU, EB')
def zhou_deng_total(data):
    """
    Computes the Zhou & Deng entropy U_ZD(m)

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing belief, plausibility, and mass values.

    Returns
    -------
    float
        The Zhou & Deng entropy value.
    """
    data = data[data.card == 1]
    additional_term = np.sum(data["plausibility"] - data["belief"])

    return zhou_deng_discord(data) + additional_term


@namer("Deng et al. (UDpe)")
@period(4)
@classification('TU, EB')
def deng_et_al(data):
    """
    Computes the Deng et al. plausibility-based entropy U(m).

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing plausibility values.

    Returns
    -------
    float
        The custom uncertainty measure U(m).
    """
    data = data[data.card == 1]
    
    total_plausibility = np.sum(data["plausibility"])

    if total_plausibility == 0:
        return 0.0

    entropy = 0.0
    for _, row in data.iterrows():
        pl_i = row["plausibility"]
        sum_pl_diff = total_plausibility - pl_i

        if sum_pl_diff > 0:
            prob = sum_pl_diff / total_plausibility
            entropy -= sum_pl_diff * np.log2(prob)

    return entropy


@namer("Zhang, Chen & Cui (U_ZCC)")
@period(4)
@classification('TU, EB')
def zhang_chen_cui_entropy(data):
    """
    Computes the Zhang, Chen & Cui entropy U_ZCC(m) of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing mass (m), belief (Bel), and plausibility (Pl) values.

    Returns
    -------
    float
        The Zhou, Chen & Cui entropy value.
    """

    def compute_dM(A, data):
        """ Compute d_M(A) based on the given formula """
        data = data[data.mass > 0]
        mass_A = data.loc[A, "mass"]

        if mass_A == 1:
            return 1
        
        # Compute sum over B ⊆ A
        sum_dM = 0
        for _, row in data.iterrows():
            B = row["element"]
            if np.all(B <= data.loc[A, "element"]):  # Check B ⊆ A
                bel_B = row["belief"]
                pl_B = row["plausibility"]
                mass_B = row["mass"]
                
                if mass_B != 1:  
                    term = ((bel_B - 0) ** 2 + (pl_B - 1) ** 2 - 1) / (2 * (mass_B - 1))
                    sum_dM += max(term, 1e-15)  # Avoid negative or zero values

        return max(sum_dM, 1e-15)  # Ensure dM(A) never reaches zero

    entropy = 0.0
    for i, row in data.iterrows():
        A = i 
        mass_A = row["mass"]
        card_A = row["card"]

        if mass_A > 0 and card_A > 0:
            dM_A = compute_dM(A, data)

            entropy -= mass_A * np.log2(dM_A / (2 ** card_A - 1))

    return entropy

@namer("Su et al. (USz)")
@period(4)
@classification('TU')
def su_et_al(data):
    """
    Computes the Su et al. (USz) entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    data = data[data.mass > 0]
    sz = 0.0
    for i in data.index:
        a = data.loc[i]
        int_sum = 0.0
        for j in data.index:
            b = data.loc[j]
            int_sum += (b.mass * (np.sum(a.element * b.element) / np.sum((a.element + b.element) > 0))) / (2**data.loc[i].card - 1)
        sz += a.mass * np.log2(int_sum)
    return -sz

@namer("Deng et al. (DXD)")
@period(4)
@classification('TU, IB')
def deng_et_al_dxd(data):
    """
    Computes the Deng and Wang entropy of the powerset.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.
    """
    dxd = 0.0
    for i in data.index:
        if np.sum(data.loc[i].element) == 1:
            dxd += 1 - np.sqrt((data.loc[i].belief)**2 + (data.loc[i].plausibility) **2 + \
                1 - 2 * data.loc[i].plausibility)
    return dxd


@namer("New Belief Entropy")
@period(4)
@classification('TU, EB')
def new_belief_entropy(data):
    """
    Computes the new belief entropy (HN) for a mass function.

    HN(m) = Σ_{A⊆Θ} m(A) log₂(2^|A| - 1) - Σ_{κ∈Θ} BPT(κ) log₂(BPT(κ))

    where BPT is a normalized belief-plausibility transformation for singletons.

    Parameters
    ----------
    data: (pandas DataFrame)
        A `pandas.DataFrame` containing the mass, belief, and plausibility
        values of the elements of a powerset.

    Returns
    -------
    float
        The new belief entropy value.
    """
    # First term: Σ m(A) * log₂(2^|A| - 1)
    focal_sets = data[data.mass > 0] 
    term1 = np.sum(focal_sets.mass * np.log2(np.power(2, focal_sets.card) - 1))

    # Second term: - Σ BPT(κ) * log₂(BPT(κ)) for singletons κ
    singletons = data[data.card == 1]
    #term1 = np.sum(singletons.mass * np.log2(np.power(2, singletons.card) - 1))
    # Calculate unnormalized BPT for singletons
    # BPT_un(κ) = (Bel(κ) + Pl(κ))/2 + (Pl(κ) - Bel(κ)) = 1.5*Pl(κ) - 0.5*Bel(κ)
    bpt_unnormalized = 1.5 * singletons.plausibility - 0.5 * singletons.belief

    total_bpt = np.sum(bpt_unnormalized)

    term2 = 0.0
    if total_bpt > 0:
        bpt_normalized = bpt_unnormalized / total_bpt
        # Calculate Shannon entropy for the BPT distribution
        term2 = -np.sum(bpt_normalized[bpt_normalized > 0] * np.log2(bpt_normalized[bpt_normalized > 0]))

    return term1 + term2

if __name__ == "__main__":
    pass
