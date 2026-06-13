#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import os
from datetime import datetime
import numpy as np
import pandas as pd
import sys
sys.path.append('..')
from evidence_theory import core
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.repair import Repair
from pymoo.core.problem import Problem
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.visualization.scatter import Scatter
from pymoo.indicators.hv import Hypervolume
from pymoo.core.population import Population
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
from itertools import chain
from itertools import combinations


def is_subset(a, b):
    """Ensure both a and b are tuples and check if a is a subset of b."""
    return set(a).issubset(set(b))

def bel2mass(powerset, belief):
    """Convert belief function values to mass function values."""
    n = len(powerset)
    mass = np.zeros(n)
    for i, A in enumerate(powerset):
        mass[i] = 0  # Initialize mass for each subset
        for j, B in enumerate(powerset):
            if is_subset(B, A): #if b is subset of a
                mass[i] += (-1)**(len(A) - len(B)) * belief[j] #calculate the mass
        mass[i] = np.round(mass[i], decimals=2) 
    return mass


def create_binary_representation(subset, frame_of_discernment):
    """Create a binary representation of a subset."""
    binary_rep = np.array([1.0 if element in subset else 0.0 for element in frame_of_discernment]) #create binary representation for using the same approach of core.py
    return binary_rep

class BeliefFunctionOptimization(Problem):
    """
    Define the multi-objective belief function optimization problem.
    """
    def __init__(self, X, entropy_measures):
        # Initialize problem parameters
        self.X = list(X) #Frame of discernment
        self.powerset_X = self.powerset(X) #Powerset of the frame of discernment
        self.n_subsets = len(self.powerset_X)  # Number of subsets in the powerset
        self.entropy_measure = entropy_measures # Entropy measure to use

        # Constraint calculation:
        disjoint_subset_pairs = [] #Store disjoint pairs for belief constraint.
        for subset1 in self.powerset_X:
            for subset2 in self.powerset_X:
                if set(subset1).isdisjoint(subset2):
                    disjoint_subset_pairs.append((subset1, subset2))

        # Number of each type of constraints:
        n_monotonicity_constraints = self.n_subsets # Monotonicity: bel1 <= bel2

        n_belief_constraints = len(disjoint_subset_pairs) * 2  # For bel1 and bel2
        n_mass = 2 * (2 ** len(self.X))  # 2 constraints for each subset in the powerset
        n_boundary = 8



        #Total number of constraints:  Make absolutely sure this is right!
        n_constr = (
            n_monotonicity_constraints +
            n_belief_constraints+
            n_mass+
            n_boundary

        )


        # Call the superclass constructor to define the problem
        super().__init__(
            n_var=self.n_subsets * 2,  # Number of variables (belief values for bel1 and bel2)
            n_obj=3,  # Number of objectives
            n_constr=n_constr,  # Number of constraints
            xl=0,  # Lower bound of belief values
            xu=1,  # Upper bound of belief values
        )

    def powerset(self, iterable):
        """Generate the powerset of a set."""
        s = list(iterable) #Convert to list
        empty_set = tuple() #Include empty set
        other_sets = list(chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))) # Generate all combinations excluding emptyset
        return [empty_set] + other_sets #Combine emptyset and other sets


    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate the objectives and constraints for a set of belief functions."""

        n_eval = x.shape[0] #Number of belief functions to evaluate
        n_subsets = self.n_subsets
        bel1 = x[:, :self.n_subsets] #Extract belief values for bel1
        bel2 = x[:, self.n_subsets:] #Extract belief values for bel2

        current_measure = self.entropy_measure #Entropy measure to use


        f1_values = [] #Store f1 values
        f2_values = [] #Store f2 values
        f3_values = [] #Store f3 values

        g_values = []  # Combined constraint array

        for i in range(n_eval):

            individual_g = []  # Constraints for this individual ONLY

            # 1. Disjoint subsets:
            g_disjoint_subset = []
            for bel in [bel1[i], bel2[i]]:
                for idx1, subset1 in enumerate(self.powerset_X):
                    for idx2, subset2 in enumerate(self.powerset_X):
                        if set(subset1).isdisjoint(subset2):
                            union_idx = self.powerset_X.index(tuple(sorted(set(subset1) | set(subset2))))
                            g_disjoint_subset.append(bel[idx1] + bel[idx2] - bel[union_idx])

            individual_g.extend(g_disjoint_subset)  # Extend with disjoint subset constraints

            # 2. Boundary Constraints:
            individual_g.append(bel1[i, 0])  
            individual_g.append(-bel1[i,0])
            individual_g.append(bel2[i, 0])  
            individual_g.append(-bel2[i,0])


            individual_g.append(-bel1[i, -1] + 1)  
            individual_g.append(bel1[i,-1] -1)
            individual_g.append(-bel2[i, -1] + 1)  
            individual_g.append(bel2[i,-1] -1)


            # 3. Monotonicity
            individual_g.extend(bel1[i] - bel2[i])

            # Calculate masses from repaired beliefs:
            m1_i = bel2mass(self.powerset_X, bel1[i])
            m2_i = bel2mass(self.powerset_X, bel2[i])

            # if np.any(m1_i < 0.0) or np.any(m2_i < 0.0):  # Check for negative mass values
            #     print("m2_i", m2_i)
            #     print("Belief values corresponding to negative m2_i:")
            #     print("bel2:", bel2[i])



            # 4. Mass non-negativity
            individual_g.extend(-m1_i)  # Use extend for 1D arrays
            individual_g.extend(-m2_i)

            # Append the INDIVIDUAL'S constraints as a 1D NumPy array:
            g_values.append(np.array(individual_g))

            frame_of_discernment = sorted(list(problem.X))

            # Create DataFrames for entropy calculations:
            data1 = pd.DataFrame({
                "mass": m1_i,
                "element": [create_binary_representation(subset, frame_of_discernment) for subset in self.powerset_X],
                "card": [len(s) for s in self.powerset_X],
                'belief': bel1[i],
                'plausibility': core.mass2plausibility(self.powerset_X, m1_i), #Call the function from core
                'commonality': core.commonality(self.powerset_X, m1_i) #Call the function from core
            })
            data2 = pd.DataFrame({
                "mass": m2_i,
                "element": [create_binary_representation(subset, frame_of_discernment) for subset in self.powerset_X],
                "card": [len(s) for s in self.powerset_X],
                'belief': bel2[i],
                'plausibility': core.mass2plausibility(self.powerset_X, m2_i),#Call the function from core
                'commonality': core.commonality(self.powerset_X, m2_i)#Call the function from core
            })

            # Calculate f3:
            F1 = [A for j, A in enumerate(self.powerset_X) if m1_i[j] > 0]  # Focal elements of m1
            F2 = [A for j, A in enumerate(self.powerset_X) if m2_i[j] > 0]  # Focal elements of m2
            f3 = len(set(F1) | set(F2)) #Union of focal elements
            f3_values.append(f3)

            f1_values.append(self.calculate_f1(data1, data2, current_measure)) #Calculate f1
            f2_values.append(self.calculate_f2(bel1[i], bel2[i])) #Calculate f2

        # Set the objectives and constraints:
        out["F"] = np.column_stack([-np.array(f1_values).reshape(-1, 1), -np.array(f2_values).reshape(-1, 1), np.array(f3_values).reshape(-1, 1)]) #Objectives (minimize)
        out["G"] = np.array(g_values)


    def calculate_f1(self, df1, df2, entropy_measure):
        """Calculate the first objective f1 (difference of entropies)."""
        uncertainty1 = entropy_measure(df1)
        uncertainty2 = entropy_measure(df2)

        if np.isclose(uncertainty1, uncertainty2, atol=1e-15):  #to avoid errors due to float precision
            return 0.0 
        else:
            return uncertainty2 - uncertainty1


    def calculate_f2(self, bel1_i, bel2_i):
        """Calculate the second objective f2 (sum of belief differences for all proper subsets).""" # Corrected description
        n = len(self.X)
        all_indices = list(range(2**n))  # All possible indices (including empty and full set)

        # Exclude indices of empty set and full set (index 0 and 2**n-1):
        indices = all_indices[1:-1] # this is correct!

        try:
            f2 = np.sum(bel2_i[indices] - bel1_i[indices])  # Sum of differences (correct calculation)
        except IndexError:
            f2 = 0  # Handle potential IndexError (though less likely with this correction)

        return f2




class QuantizedLHS(LHS):
    """Latin Hypercube Sampling with quantization."""
    def _do(self, problem, n_samples, **kwargs):
        samples = super()._do(problem, n_samples, **kwargs) #Call standard LHS from pymoo
        return np.round(samples * 100) / 100.0  # Quantize samples


def generate_initial_beliefs(problem):
    """Generates initial belief function values for warm start."""
    n_subsets = problem.n_subsets
    X = problem.X  # Frame of discernment

    bel1 = np.zeros(n_subsets)
    bel1[-1] = 1.0  # Vacuous belief

    bel2 = np.zeros(n_subsets)
    try:
        singleton1_index = problem.powerset_X.index((1,))
        singleton2_index = problem.powerset_X.index((2,))

        if len(X) >= 2:
            y = 2 ** len(problem.X)
            bel2[y-2] = 0.1
            #bel2[y-1] = 0.68
            bel2[-1] = 1
        elif len(X) == 1:
            bel2[singleton1_index] = 1.0  

    except ValueError:
        #print("ERROR: Singletons (1,) or (2,) not found in powerset.")
        return None

    return np.concatenate([bel1, bel2])



class QuantizedWarmStartLHS(QuantizedLHS):
    def _do(self, problem, n_samples, **kwargs):
        # Generate initial beliefs using `generate_initial_beliefs`
        warm_start_sample = generate_initial_beliefs(problem)

        if warm_start_sample is not None:
            # Expand to a population of warm starts if needed
            warm_start_population = np.tile(warm_start_sample, (n_samples, 1))
            warm_start_population = np.round(warm_start_population * 100) / 100.0  # Quantize to {0, 0.01, ..., 1}

        else:
            warm_start_population = super()._do(problem, n_samples, **kwargs)  # Fall back to standard sampling

        # Combine warm start with random sampling
        lhs_samples = super()._do(problem, n_samples, **kwargs)
        combined_population = np.vstack([warm_start_population, lhs_samples])
        return combined_population[:n_samples]  # Ensure the population matches `n_samples`


class BeliefRepair(Repair):
    def _do(self, problem, X, **kwargs):
        n_subsets = problem.n_subsets

        X[:, :n_subsets][:, 0] = 0.0  # bel({}) = 0 for bel1
        X[:, n_subsets:][:, 0] = 0.0  # bel({}) = 0 for bel2
        X[:, :n_subsets][:, -1] = 1.0  # bel(X) = 1 for bel1
        X[:, n_subsets:][:, -1] = 1.0  # bel(X) = 1 for bel2

        max_iterations = 5

        for _ in range(max_iterations):

            changed = False

            # Set bel({}) and bel(X):
            X[:, :n_subsets][:, 0] = 0.0  # bel({}) = 0 for bel1
            X[:, n_subsets:][:, 0] = 0.0  # bel({}) = 0 for bel2
            X[:, :n_subsets][:, -1] = 1.0  # bel(X) = 1 for bel1
            X[:, n_subsets:][:, -1] = 1.0  # bel(X) = 1 for bel2

            #Disjoint Subset Constraint (using accumulated updates):
            for bel_index in range(2):
                bel = X[:, n_subsets * bel_index : n_subsets * (bel_index + 1)].copy()
                updated_bel = bel.copy()  
                for subset1_index, subset1 in enumerate(problem.powerset_X):
                    for subset2_index, subset2 in enumerate(problem.powerset_X):
                        if set(subset1).isdisjoint(subset2):
                            union_idx = problem.powerset_X.index(tuple(sorted(set(subset1) | set(subset2))))
                            current_sum = bel[:, subset1_index] + bel[:, subset2_index]
                            updated_bel[:, union_idx] = np.maximum(updated_bel[:, union_idx], current_sum)

                if np.any(updated_bel != bel):
                    X[:, n_subsets * bel_index : n_subsets * (bel_index + 1)] = updated_bel
                changed = True




            # Singleton Sum Constraint (bel2 only):
            bel2 = X[:, n_subsets:].copy()  #Extract bel2
            singleton_indices = [problem.powerset_X.index((x,)) for x in problem.X if (x,) in problem.powerset_X]
            singleton_sum = np.sum(bel2[:, singleton_indices], axis=1, keepdims=True)
            for i in range(X.shape[0]):
                if singleton_sum[i] > 1:
                    excess = singleton_sum[i] - 1

                    for singleton_index in singleton_indices:
                        reduction = min(bel2[i, singleton_index], excess) # Subtract at most the belief value or the excess
                        bel2[i, singleton_index] -= reduction
                        excess -= reduction  # Update the remaining excess
                        if excess <= 1e-9: # Exit if excess is small enough
                            break
                        changed = True




            X[:, problem.n_subsets:] = bel2 #Update bel2 on X

            # Belief Sum Constraint (bel2 only):
            bel2 = X[:, n_subsets:].copy()
            for i in range(X.shape[0]):
                bel_tot = 0
                for idx, A in enumerate(problem.powerset_X[1:-1]):
                    subset_index = problem.powerset_X.index(A)
                    assignable_belief = bel2[i, subset_index] - sum(
                        bel2[i, problem.powerset_X.index(B)] for B in problem.powerset_X if set(B).issubset(A) and B != A
                    )
                    bel_tot += assignable_belief

                    if bel_tot > 1:
                        excess = bel_tot - 1
                        #print("excess", excess)

                        # Directly subtract the excess from the current subset's belief:
                        bel2[i, subset_index] -= excess 
                        #print("bel2 new", bel2[i, subset_index])

                        bel_tot = 1  # Reset bel_tot
                        changed = True
            X[:, problem.n_subsets:] = bel2  # Update bel2 in X  

            if not changed:
                break

        # Monotonicity constraint (between bel1 and bel2) AFTER other repairs:
        bel1 = X[:, :n_subsets]
        bel2 = X[:, n_subsets:]
        X[:, :n_subsets] = np.minimum(bel1, bel2) # bel1 &lt;= bel2

        # Quantization (AFTER all repairs):
        X = np.round(X * 100) / 100.0
        return X




def process_solution_for_entropy(solution, problem, frame_of_discernment, uncertainty_measure):
    """Process a solution and calculate entropies."""

    n_subsets = problem.n_subsets
    bel1 = solution[:n_subsets] #Extract bel1
    bel2 = solution[n_subsets:] #Extract bel2

    #Calculate masses:
    m1 = bel2mass(problem.powerset_X, bel1) 
    m2 = bel2mass(problem.powerset_X, bel2)

    # Calculate plausibilities and commonalities (using functions from 'core' module):
    pl1 = core.mass2plausibility(np.array(problem.powerset_X, dtype=object), m1)
    pl2 = core.mass2plausibility(np.array(problem.powerset_X, dtype=object), m2)
    comm1 = core.commonality(problem.powerset_X, m1)
    comm2 = core.commonality(problem.powerset_X, m2)

    #Create dataframes for entropy calculations:
    data1 = pd.DataFrame({
        "mass": m1,
        "element": [create_binary_representation(subset, frame_of_discernment) for subset in problem.powerset_X],
        "card": [len(s) for s in problem.powerset_X],
        "belief": bel1,
        "plausibility": pl1,
        "commonality": comm1
    })
    data2 = pd.DataFrame({
        "mass": m2,
        "element": [create_binary_representation(subset, frame_of_discernment) for subset in problem.powerset_X],
        "card": [len(s) for s in problem.powerset_X],
        "belief": bel2,
        "plausibility": pl2,
        "commonality": comm2
    })


    uncertainty1 = uncertainty_measure(data1) # Calculate entropy for bel1
    uncertainty2 = uncertainty_measure(data2) #Calculate entropy for bel2

    # Create and print belief and mass dataframes:
    belief_df = pd.DataFrame({'Bel1': bel1, 'Bel2': bel2}, index=[str(s) for s in problem.powerset_X])
    mass_df = pd.DataFrame({'m1': m1, 'm2': m2}, index=[str(s) for s in problem.powerset_X])

    print("Belief Functions:")
    print(belief_df)
    print("\nMass Functions:")
    print(mass_df)

    return uncertainty1, uncertainty2, bel1, m1, bel2, m2



# Example usage (main optimization loop)

entropy_measures = [
#core.hohle,
# # core.yager,
# # core.yager_non_specificity,
# # core.smets,
# # core.dubois_prade,
# # core.nguyen,
# # core.dubois_prade_commonality,
# # core.dubois_prade_fuzziness,
# # core.dubois_prade_imprecision,
# # core.lamata_moral,
# # core.lamata_moral_upper,
# # core.klir_and_ramer,
# # core.klir_total_uncertainty,
core.harmanec_and_klir, #FATTA
# # core.klir_and_parviz,
# # core.pal_et_al,
# # core.maeda_hichihashi,   
# # core.george_and_pal,
# # core.maluf, 
# # core.klir_shannon,
# # core.yager_shapley,
# # core.jousselme_et_al,
# # core.yang_and_han,
# # core.deng,
#core.wang_and_song,
# # core.zhou_et_al,
# core.tang, #FATTA
#core.pan_and_deng,
# # core.jirousek_and_shenoy,
# core.jirousek_shenoy_q_entropy, #FATTA
# # core.mambe,
# # core.cui_et_al,
# core.li, #FATTA
# # core.pan_2nd_entropy,
#core.chen_et_al_cds, #FATTA
#core.gao,
#core.zhao_entropy, #FATTA
#core.yan_and_deng, #FATTA
#core.qin_et_al, #FATTA
# # core.li_and_pan,
#core.li_improved,
core.wen_entropy,
#core.chen_improved_c, #FATTA
core.deng_and_wang,
#core.dezert_tchamova_betp,
#core.zhang,
#core.li_et_al,
# core.fractal_based_entropy, #FATTA
# core.zhou_belief_entropy, #FATTA
#core.xue_deng_entropy, #FATTA
#core.dutta_and_shome, #FATTA
#core.new_chen_deng, #FATTA
core.cui_and_deng,
# # core.kavya_et_al,
#core.zhou_deng_discord, #FATTA
# # core.zhou_deng_total,
#core.deng_et_al,
# core.zhang_chen_cui_entropy #FATTA
# core.ram
#core.su_et_al
#core.deng_et_al_dxd
core.new_belief_entropy
]



parser = argparse.ArgumentParser(description="NSGA-II belief function optimization")
parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
parser.add_argument("--n-gen", type=int, default=200, help="Number of generations (default: 200)")
parser.add_argument("--pop-size", type=int, default=200, help="Population size (default: 200)")
parser.add_argument("--fod-size", type=int, default=2, help="Maximum frame of discernment size (default: 2)")
args = parser.parse_args()

_run_date = datetime.now().strftime("%Y%m%d")
_run_id = f"{args.n_gen}gen_{args.pop_size}pop_seed{args.seed}_{_run_date}"
RESULTS_ROOT = "results"

max_fod_size = args.fod_size  #Maximum size of the frame of discernment
results = []  # Store the results of each optimization run
monotonicity_violations = {}  # Store monotonicity violations
all_measures_data = []

entropy_measures_to_check = entropy_measures.copy() #Create a copy of the list for processing

for fod_size in range(2, max_fod_size + 1): #Iterate over frame of discernment sizes
    print(f"\n\nFOD Size: {fod_size}")
    X = set(range(1, fod_size + 1)) # Frame of discernment
    pop_size = args.pop_size  #Population size
    n_gen = args.n_gen  #Number of generations

    measures_for_current_fod = entropy_measures_to_check.copy() #List of measures for the current FOD size

    for measure in measures_for_current_fod.copy(): #Iterate over entropy measures
        print(f"Starting simulation for entropy measure: {measure.__name__}")
        problem = BeliefFunctionOptimization(X, entropy_measures=measure) #Create pymoo problem instance

        # Configure the NSGA-II algorithm:
        algorithm = NSGA2(pop_size=pop_size, sampling=QuantizedWarmStartLHS(QuantizedLHS), repair=BeliefRepair()) #

        # Run the optimization
        res = minimize(problem, algorithm, ('n_gen', n_gen), verbose=True, save_history=True, seed=args.seed)


        if res.F is not None and res.F.size > 0: #Check if feasible solutions where found
            frame_of_discernment = sorted(list(problem.X))

            print(f"\n{'='*80}")
            print(f"ANALISI DETTAGLIATA FRONTE DI PARETO - Misura: {measure.__name__}")
            print(f"{'='*80}")

            all_pareto_details = []

            # Iteriamo su tutte le soluzioni del fronte non dominato
            for i in range(len(res.X)):
                sol = res.X[i]
                # Calcoliamo i valori reali degli obiettivi (invertendo il segno dove necessario)
                f1_current = -res.F[i, 0] # Δ Entropy (u2 - u1)
                f2_current = -res.F[i, 1] # Δ Belief
                f3_current = int(res.F[i, 2]) # Focal Elements

                print(f"\n>>> ANALISI PUNTO PARETO #{i+1} [f1={f1_current:.6f}, f2={f2_current:.2f}, f3={f3_current}]")

                # Calcolo dettagliato e stampa tabelle (tramite la tua funzione esistente)
                u1, u2, b1, m1, b2, m2 = process_solution_for_entropy(
                    sol, problem, frame_of_discernment, measure
                )

                is_monotone = u1 >= (u2 - 1e-12)

                detail = {
                    'index': i, 'f1': f1_current, 'f2': f2_current, 'f3': f3_current,
                    'u1': u1, 'u2': u2, 'monotone': is_monotone
                }
                all_pareto_details.append(detail)

                # Focus su f3=2 e violazioni
                if f3_current == 2 and not is_monotone:
                    print(f"!!! ALERT: Violazione trovata con f3=2 (struttura semplice) !!!")

                if not is_monotone:
                    print(f"!!! MONOTONICITY VIOLATED: Uncertainty1 ({u1:.6f}) < Uncertainty2 ({u2:.6f})")
                else:
                    print(f"Monotonicity Verified.")

                print("-" * 40)

            # --- Identificazione del Punto Medio ---
            # Ordiniamo i risultati per f1 per trovare il punto centrale
            sorted_pareto = sorted(all_pareto_details, key=lambda x: x['f1'])
            mid_idx = len(sorted_pareto) // 2
            punto_medio = sorted_pareto[mid_idx]

            print(f"\n{'*'*20} PUNTO MEDIO DEL FRONTE {'*'*20}")
            print(f"Il punto medio (mediana di f1) è il Punto #{punto_medio['index']+1}:")
            print(f"f1: {punto_medio['f1']:.6f}, f2: {punto_medio['f2']:.2f}, f3: {punto_medio['f3']}")
            print(f"Incertezze: U1={punto_medio['u1']:.6f}, U2={punto_medio['u2']:.6f}")
            print(f"{'*'*60}\n")


            # Extract best solutions for each objective
            best_f1_index = np.argmin(res.F[:, 0]) #Best solution according to f1
            best_f2_index = np.argmin(res.F[:, 1]) #Best solution according to f2
            best_f3_index = np.argmin(res.F[:, 2]) #Best solution according to f3
            best_solutions = [res.X[best_f1_index], res.X[best_f2_index], res.X[best_f3_index]] #Store best solutions
            best_labels = ["Best f1", "Best f2", "Best f3"] #Store corresponding labels

        # Check for negative masses AFTER optimization:
            for i in range(len(res.X)):
                bel2 = res.X[i, problem.n_subsets:]
                m2 = bel2mass(problem.powerset_X, bel2)
                if np.any(m2 < 0):
                    print(f"Negative mass values found for {measure.__name__} in solution {i+1}:")
                    print(m2)
                    # Optionally, print the corresponding belief values:
                    print("Corresponding bel2 values:")
                    print(bel2)

        else: #Handle the case with no feasible solutions
            print(f"No feasible solutions found for {measure.__name__} at FOD size {fod_size}")
            continue #Skip to the next entropy measure

        #Process best solutions:
        for solution, label in zip(best_solutions, best_labels):
            print(f"Solution for {label}, {measure.__name__}:")

            #Calculate entropy and extract belief/mass functions:
            uncertainty1, uncertainty2, bel1, m1, bel2, m2 = process_solution_for_entropy(
                solution, problem, frame_of_discernment, measure
            )

            print(f"  Uncertainty1: {uncertainty1:.10f}")
            print(f"  Uncertainty2: {uncertainty2:.10f}")

            # Check for monotonicity:
            is_monotone = uncertainty1 >= uncertainty2 

            if not is_monotone: #Monotonicity violated
                print("  Monotonicity Violated")

                #Store violation data
                violation_data = {
                    'FOD': fod_size,
                    'bel1': bel1,
                    'm1': m1,
                    'bel2': bel2,
                    'm2': m2,
                    'uncertainty1': uncertainty1,
                    'uncertainty2': uncertainty2,
                }
                monotonicity_violations[measure.__name__] = violation_data #Store violation information
                if measure in entropy_measures_to_check: #Remove violated measure
                    entropy_measures_to_check.remove(measure) #Remove measure so it won't be retested in larger FODs
            else:
                print("  Monotonicity Verified")

            # Extract and print F-Values:
            if label == "Best f1":
                f_value = -res.F[best_f1_index, 0] #changed sign
            elif label == "Best f2":
                f_value = -res.F[best_f2_index, 1] #changed sign
            elif label == "Best f3":
                f_value = res.F[best_f3_index, 2]


            result_data = [label, measure.__name__, uncertainty1, uncertainty2, f_value, bel1, m1, bel2, m2, fod_size, is_monotone] #Store results in a list
            results.append(result_data)

            print(f"  {label[5:]} Value: {f_value}")
            print("-" * 20)



        #Hypervolume calculation:
        # # Hypervolume calculation with normalization
        # max_f = np.zeros(3)  # Array for storing max objective values (for hypervolume reference)
        # f_min = np.inf * np.ones(3)  # Initialize with high values
        # f_max = -np.inf * np.ones(3)  # Initialize with low values

        # for generation in res.history:
        #     gen_F = generation.pop.get("F")
        #     f_min = np.minimum(f_min, np.min(gen_F, axis=0))
        #     f_max = np.maximum(f_max, np.max(gen_F, axis=0))

        # # Normalize objective values before calculating hypervolume
        # normalized_F = (res.F - f_min) / (f_max - f_min + 1e-6)

        # # Set a normalized reference point just beyond the observed maximum
        # reference_point = np.array([1.1, 1.1, 1.1])


        # hv = Hypervolume(ref_point=reference_point, normalize=False) #Hypervolume indicator instance
        # hv_values = [] #Store hypervolume values across generations
        # for generation in res.history:
        #     if generation.opt is not None: #Check if the generation has optimal solutions
        #         F = generation.opt.get("F") #Get objective values
        #         if F.ndim == 1:  #Handle single solution case
        #             F = F.reshape(1, -1)

        #         # Normalize objective values:
        #         F_normalized = F.copy()
        #         for i in range(F.shape[1]): #Iterate over objectives
        #             range_ = np.max(F[:, i]) - np.min(F[:, i])
        #             if range_ > 1e-6: #Avoid division by zero
        #                 F_normalized[:, i] = (F[:, i] - np.min(F[:, i])) / range_

        #         hv_value = hv.do(F_normalized) #Calculate normalized hypervolume
        #         hv_values.append(hv_value)

        #     else:  #No solutions in this generation
        #         hv_values.append(0)

        # # Plot hypervolume convergence:
        # if hv_values: #Check if hypervolume is available to plot
        #     fig, ax = plt.subplots()
        #     ax.plot(np.arange(len(res.history)), hv_values) #Hypervolume over generations
        #     ax.set_xlabel("Generation", fontsize=16)
        #     ax.set_ylabel("Hypervolume", fontsize=16)
        #     ax.set_title(f"Convergence Plot - Hypervolume - {measure.__name__}", fontsize=18)
        # Hypervolume calculation:
        # Hypervolume calculation with normalization
        max_f = np.zeros(3)  # Array for storing max objective values
        f_min = np.inf * np.ones(3)  # Initialize with high values
        f_max = -np.inf * np.ones(3)  # Initialize with low values

        # 1. First pass: find global min/max across all generations for consistent normalization
        for generation in res.history:
            gen_F = generation.pop.get("F")
            f_min = np.minimum(f_min, np.min(gen_F, axis=0))
            f_max = np.maximum(f_max, np.max(gen_F, axis=0))

        # Set a normalized reference point just beyond the observed maximum
        reference_point = np.array([1.1, 1.1, 1.1])
        hv = Hypervolume(ref_point=reference_point, normalize=False) 

        hv_values = [] # For plotting
        hv_generation_data = [] # List to store data for saving
        measure_name = measure.__name__

        # 2. Second pass: Calculate HV per generation using global normalization
        for i, generation in enumerate(res.history):
            if generation.opt is not None: 
                F = generation.opt.get("F")
                if F.ndim == 1:
                    F = F.reshape(1, -1)

                # Normalize objective values using the GLOBAL bounds
                # This ensures HV values are comparable across generations
                F_normalized = (F - f_min) / (f_max - f_min + 1e-6)

                hv_value = hv.do(F_normalized)
                hv_values.append(hv_value)
            else:
                hv_value = 0
                hv_values.append(hv_value)

            # Store the record for this generation
            hv_generation_data.append({
                'measure': measure_name,
                'generation': i,
                'hypervolume': hv_value
            })

        # --- SAVING THE DATA ---
        measure_dir = os.path.join(RESULTS_ROOT, measure_name)
        os.makedirs(measure_dir, exist_ok=True)

        hv_df = pd.DataFrame(hv_generation_data)
        csv_filename = os.path.join(measure_dir, f"hv_convergence_{measure_name}_fod{fod_size}_{_run_id}.csv")
        hv_df.to_csv(csv_filename, index=False)
        print(f"Hypervolume data saved to {csv_filename}")

        # Plot hypervolume convergence:
        if hv_values:
            fig, ax = plt.subplots()
            ax.plot(np.arange(len(res.history)), hv_values)
            ax.set_xlabel("Generation", fontsize=16)
            ax.set_ylabel("Hypervolume", fontsize=16)
            ax.set_title(f"Convergence Plot - Hypervolume - {measure_name}", fontsize=18)
            # plt.show()
        measure_name = measure.__name__

        # Estrazione dei valori dagli output dell'ottimizzazione
        f1 = -res.F[:, 0]  # Entropy Difference
        f2 = -res.F[:, 1]  # Belief Difference, reso positivo
        f3 = res.F[:, 2]  # Numero di elementi focali

        # Creazione del grafico
        fig, ax = plt.subplots(figsize=(8, 6))

        # Scatter plot con colore determinato da f3
        scatter = ax.scatter(f1, f2, c=f3, cmap=cm.viridis, s=50, edgecolor='k', alpha=0.8)

        # Creazione di una griglia regolare per interpolare f3
        grid_x, grid_y = np.meshgrid(
            np.linspace(f1.min(), f1.max(), 100),
            np.linspace(f2.min(), f2.max(), 100)
        )

        # Scatter plot con colori distinti per f3 senza barra di colore
        unique_f3 = np.unique(f3)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_f3)))

        # Interpolazione dei valori di f3 (uso nearest per evitare artefatti)
        grid_z = griddata((f1, f2), f3, (grid_x, grid_y), method='nearest')

        # Scatter plot con colori distinti per f3 senza barra di colore
        unique_f3 = np.unique(f3)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_f3)))

        for i, f3_val in enumerate(unique_f3):
            mask = f3 == f3_val
            ax.scatter(f1[mask], f2[mask], color=colors[i], label=f'$f_3={int(f3_val)}$', s=50, edgecolor='k', alpha=0.8)

        # Etichette e titolo
        ax.set_xlabel(r'$f_1$ ', fontsize=14)
        ax.set_ylabel(r'$f_2$ ', fontsize=14, rotation=0)
        ax.set_title(f"Objective Space ", fontsize=16)

        # Aggiunta della legenda
        ax.legend(fontsize=14)

        # Dimensioni del testo negli assi
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Aggiunta della linea tratteggiata per f1 = 0
        ax.axvline(x=0, color='gray', linestyle='dashed')

        pdf_filename = os.path.join(measure_dir, f"pareto_plot_{measure_name}_fod{fod_size}_{_run_id}.pdf")
        plt.savefig(pdf_filename, format="pdf")

        plt.tight_layout()
        # plt.show()


        # Store the results in the list
        for i in range(len(f1)):
            all_measures_data.append({
                'measure': measure_name,
                'f1': f1[i],
                'f2': f2[i],
                'f3': f3[i]
            })

    # Convert the accumulated list to a DataFrame **after the loop**
    df = pd.DataFrame(all_measures_data)

    os.makedirs(RESULTS_ROOT, exist_ok=True)
    csv_file_path = os.path.join(RESULTS_ROOT, f"measures_results_fod{fod_size}_{_run_id}.csv")
    df.to_csv(csv_file_path, index=False)

    print(f"Results saved to {csv_file_path}")





# Create and save results dataframe:
results_df = pd.DataFrame(results, columns=["Label", "Measure", "Uncertainty1", "Uncertainty2", "F_Value", "Bel1", "m1", "Bel2", "m2", "FOD", "Is Monotone"])
results_df.to_csv(os.path.join(RESULTS_ROOT, f"all_results_fod{max_fod_size}_{_run_id}.csv"), index=False)


#Print detailed violations recap:
print("\nMonotonicity Violations Recap:")

#Group results by measure for detailed violations analysis:
results_by_measure = results_df.groupby("Measure") #Grouped by measure name
for measure_name, measure_group in results_by_measure:
    if not measure_group["Is Monotone"].all(): #Check if any run of this measure has violations
        print(f"Measure: {measure_name}")

        violation_rows = measure_group[~measure_group["Is Monotone"]]  # Filter rows where monotonicity was violated


        for _, row in violation_rows.iterrows():  # Iterate through violation rows
            print(f"Solution for {row['Label']}, {measure_name}:") # Label of the best solution

            # Recreate powerset based on the specific FOD size
            current_fod_size = row['FOD']
            X_current = set(range(1, current_fod_size + 1))
            powerset_X_current = problem.powerset(X_current) #Use problem's powerset method for consistency

            # Create belief and mass dataframes:
            belief_df = pd.DataFrame({'Bel1': row['Bel1'], 'Bel2': row['Bel2']}, index=[str(s) for s in powerset_X_current])
            mass_df = pd.DataFrame({'m1': row['m1'], 'm2': row['m2']}, index=[str(s) for s in powerset_X_current])

            print("Belief Functions:")
            print(belief_df)
            print("\nMass Functions:")
            print(mass_df)

            print(f"  Uncertainty1: {row['Uncertainty1']:.10f}")
            print(f"  Uncertainty2: {row['Uncertainty2']:.10f}")
            print("  Monotonicity Violated")  #We know it's violated here
            print(f"  {row['Label'][5:]} Value: {row['F_Value']}")
            print("-" * 20)



# Create and save monotonicity summary dataframe
monotonicity_summary = [] #List to build the df
for measure in entropy_measures: #Iterate over all entropy measures
    is_monotone_overall = 'Y' #Initialize as monotone
    if measure.__name__ in monotonicity_violations: #If found in violations dictionary
        is_monotone_overall = 'N' #Mark as non-monotone
    monotonicity_summary.append([measure.__name__, is_monotone_overall])


monotonicity_df = pd.DataFrame(monotonicity_summary, columns=["Entropy Measure", "Is Monotone"])
monotonicity_df.to_csv(os.path.join(RESULTS_ROOT, f"monotonicity_recap_fod{max_fod_size}_{_run_id}.csv"), index=False)


print("\nMonotonicity Summary Table:")
print(monotonicity_df)

