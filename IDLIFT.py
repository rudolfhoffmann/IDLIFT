# -*- coding: utf-8 -*-
"""
IDLIFT — Interval-based Dynamic LIFT
Author: Rudolf Hoffmann and Christoph Reich
Based on: LIFT by M. Nauta (https://github.com/M-Nauta/LIFT).
"""
import pandas as pd
import numpy as np
import itertools
import copy
from scipy.stats import chi2, chisquare
import random


VERBOSE = 0

# ---------------------------------------------------------------------------
# Chi-squared critical value
# ---------------------------------------------------------------------------

def getChi2(confidence):
    # Get the critical value for the chi-squared distribution with 1 degree of freedom.
    # Used to assess whether a pamh score is statistically significant.
    df = 1
    critical_value = chi2.ppf(confidence, df)
    critical_value = round(critical_value, 3)
    return critical_value


# ---------------------------------------------------------------------------
# ITET cell types
#   0: event did not occur in this observation window
#   int / float: binary label column (e.g. issue = 0 or 1)
#   frozenset: set of (start, end) interval tuples (one or more occurrences)
# ---------------------------------------------------------------------------

INTERVAL_TOLERANCE = 0.05  # seconds — minimum genuine overlap required for AND; also used as the tolerance margin for PAND ordering


def get_intervals(val):
    # Return the list of (start, end) tuples stored in an ITET cell. Returns an empty list if the cell represents "did not occur" (value 0).
    if isinstance(val, (set, frozenset)): # is val a set or frozenset
        return list(val)
    return []


def to_bool(val):
    # Convert an ITET cell to a binary 0/1 value.
    # A set with at least one interval counts as 1 (event occurred).
    # A numeric value > 0 also counts as 1 (used for label columns).
    if isinstance(val, (set, frozenset)): # is val a set or frozenset
        return 1 if val else 0
    if isinstance(val, (int, float, np.integer, np.floating)): # is val an int, float, np.integer or np.floating
        return 1 if val > 0 else 0
    return 0


def _is_binary_col(col_values):
    # Check whether a column contains only plain 0 or 1 values (i.e. it is a binary label column, not an interval-valued sensor column).
    for v in col_values:
        if not isinstance(v, (int, float, np.integer, np.floating)): # is val an int, float, np.integer or np.floating
            return False
        if int(v) not in (0, 1):
            return False
    return True


def starts_before_and_overlaps(a, b, tolerance):
    """
    True if interval "a" starts clearly before "b" AND "a" is still active when "b" starts.

    Condition 1:  a[0] + tolerance < b[0]   — "a" starts strictly before "b" (not simultaneous)
    Condition 2:  a[1] + tolerance > b[0]   — "a" has not ended when "b" starts (co-active)

    Example: a = [0.0, 1.8], b = [1.5, 2.0] --> a starts before b and is still running when b starts → True
    """
    ordered_starts = a[0] + tolerance < b[0]
    co_active = a[1] + tolerance > b[0]
    return ordered_starts and co_active


def intervals_overlap(a, b, tolerance):
    # True if intervals a and b genuinely overlap by more than the tolerance. Both directions must overlap — this is a symmetric check.
    return a[0] + tolerance < b[1] and b[0] + tolerance < a[1]


def interval_intersection(a, b, tolerance):
    # Return the intersection of two intervals, or None if they do not overlap by more than the tolerance. 
    # Used by mergeAND to check genuine co-activity.
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    if e - s > tolerance:
        return (s, e)
    return None


# ---------------------------------------------------------------------------
# Gate merge functions
# Each mergeXXX function appends a new binary column to the dataframe indicating whether the gate fires in each row.
# ---------------------------------------------------------------------------

def mergeOR(df, tomerge, parent, tolerance):
    # OR fires when at least one child interval overlaps with a parent interval.
    # In rows where the parent did not fire (no intervals), fall back to a boolean presence check instead of skipping the row. 
    # Skipping parent-inactive rows makes every candidate gate score identically (all return 0), 
    # so the PAMH test cannot distinguish correct children from wrong ones.
    # The boolean fallback exposes spurious candidates: a shared BE may carry intervals in parent-inactive rows.
    dfextended = df.copy()
    tomergeindices = []
    for k in tomerge:
        index = df.columns.get_loc(k)
        tomergeindices.append(index)
    parent_index = df.columns.get_loc(parent)
    dataset      = df.values
    newcolumn    = np.zeros((len(dataset), 1))

    for i in range(len(dataset)):
        parent_ivs = get_intervals(dataset[i, parent_index])
        if parent_ivs:
            # Parent has intervals — OR fires if any child interval overlaps with any parent interval
            for idx in tomergeindices:
                for c in get_intervals(dataset[i, idx]):
                    for p in parent_ivs:
                        if intervals_overlap(c, p, tolerance):
                            newcolumn[i, 0] = 1
                            break
                    if newcolumn[i, 0] == 1:
                        break
                if newcolumn[i, 0] == 1:
                    break
        else:
            # Parent inactive — boolean presence check to expose false positives from wrong candidates
            # (correct children are guaranteed inactive when the parent is inactive)
            if any(get_intervals(dataset[i, idx]) for idx in tomergeindices):
                newcolumn[i, 0] = 1

    dfextended['OR'] = newcolumn
    return dfextended


def mergeAND(df, tomerge, tolerance):
    # AND fires when there exists at least one combination of child intervals (one per child) that all share a common intersection (genuine co-activity).
    dfextended = df.copy()
    tomergeindices=[]
    for k in tomerge:
        index=df.columns.get_loc(k)
        tomergeindices.append(index)
    dataset = df.values
    newcolumn = np.zeros((len(dataset), 1))

    for i in range(len(dataset)):
        interval_sets = [get_intervals(dataset[i, idx]) for idx in tomergeindices]

        # If any child has no intervals at all, AND cannot fire in this row
        if any(len(ivs) == 0 for ivs in interval_sets):
            continue

        # Try all combinations of one interval per child, because each child event can have multiple intervals in a single row.
        for combo in itertools.product(*interval_sets):
            intersection = combo[0]
            for iv in combo[1:]:
                intersection = interval_intersection(intersection, iv, tolerance)
                if intersection is None:
                    break  # this combination does not intersect — try the next one
            if intersection is not None:
                newcolumn[i, 0] = 1
                break  # one valid combination is enough

    dfextended['AND'] = newcolumn
    return dfextended


def mergePAND(df, tomerge, tolerance, parent=None):
    # PAND fires when there exists a combination of child intervals that are strictly ordered (each starts clearly before the next, while still active).
    # If the parent column contains interval values (not just 0/1), the last child interval must also overlap with a parent interval.
    dfextended = df.copy()
    tomergeindices=[]
    for k in tomerge:
        index=df.columns.get_loc(k)
        tomergeindices.append(index)
    parent_index = df.columns.get_loc(parent) if parent is not None else None
    dataset = df.values
    newcolumn = np.zeros((len(dataset), 1))

    parent_is_binary = (
        parent_index is not None
        and _is_binary_col(dataset[:, parent_index])
    )

    for i in range(len(dataset)):
        child_ivs = [get_intervals(dataset[i, idx]) for idx in tomergeindices]

        # All children must have at least one interval
        if any(len(ivs) == 0 for ivs in child_ivs):
            continue

        valid = False
        for combo in itertools.product(*child_ivs):
            # Check that each consecutive pair satisfies the ordering condition
            ordering_ok = all(
                starts_before_and_overlaps(combo[j], combo[j + 1], tolerance)
                for j in range(len(combo) - 1)
            )
            if not ordering_ok:
                continue

            # For non-binary parents, the last child must overlap a parent interval
            if parent_index is not None and not parent_is_binary:
                parent_ivs = get_intervals(dataset[i, parent_index])
                overlaps_parent = any(
                    intervals_overlap(combo[-1], p, tolerance) for p in parent_ivs
                )
                if not overlaps_parent:
                    continue

            valid = True
            break

        if valid:
            newcolumn[i, 0] = 1

    dfextended['PAND'] = newcolumn
    return dfextended


def mergeSEQ(df, tomerge, tolerance):
    # SEQ is stricter than PAND: All interval combinations must be ordered.
    # PAND allows out-of-order combinations; SEQ does not, but noise may cause a few violations even in SEQ data.
    # SEQ = 1 when every combination is ordered (no violation).
    # SEQ_VIOLATION = 1 when at least one combination is out of order.
    # testSEQvsPAND uses a statistical test on these counts to decide whether the observed violations exceed what noise alone would explain.
    dfextended = df.copy()
    tomergeindices=[]
    for k in tomerge:
        index=df.columns.get_loc(k)
        tomergeindices.append(index)
    dataset = df.values
    newcolumn_seq = np.zeros((len(dataset), 1))
    newcolumn_violation = np.zeros((len(dataset), 1))

    for i in range(len(dataset)):
        child_ivs = [get_intervals(dataset[i, idx]) for idx in tomergeindices]

        if any(len(ivs) == 0 for ivs in child_ivs):
            continue

        has_valid = False
        has_violation = False

        for combo in itertools.product(*child_ivs):
            ordered = all(
                starts_before_and_overlaps(combo[j], combo[j + 1], tolerance)
                for j in range(len(combo) - 1)
            )
            if ordered:
                has_valid = True
            else:
                has_violation = True

        if has_valid and not has_violation:
            newcolumn_seq[i, 0] = 1
        elif has_violation:
            newcolumn_violation[i, 0] = 1

    dfextended['SEQ'] = newcolumn_seq
    dfextended['SEQ_VIOLATION'] = newcolumn_violation
    return dfextended


# ---------------------------------------------------------------------------
# Subset generation and stratification
# ---------------------------------------------------------------------------

def getavailablesets(df, parent_node, seen_parent, subset_range=(2, 5)):
    # Collect all columns that are valid candidates to be children of parent_node.
    # Already-processed nodes (seen_parent) and the parent itself are excluded.
    # 'AND' and 'OR' are temporary column names added by mergeAND and mergeOR, so they are excluded, too.
    candidates = [
        node for node in df.columns if node not in seen_parent and node != parent_node and node not in ('AND', 'OR')
    ]

    # Generate all combinations of size subset_range[0] up to subset_range[1] - 1
    availablesets = []
    for size in range(subset_range[0], subset_range[1]):
        for subset in itertools.combinations(candidates, size):
            availablesets.append(subset)

    return availablesets


def getstratum(df, parent_node, test_attribute, attribute_values):
    # Build a 2x2 contingency table (stratum) for the PAMH test.
    # Rows: test_attribute = 0 / 1
    # Columns: parent_node = 0 / 1
    df_binary = df.copy().map(to_bool)

    # Apply any additional filtering conditions passed in attribute_values
    for key in attribute_values:
        df_binary = df_binary.loc[df_binary[key] == attribute_values[key]]

    c = np.ones((2, 2))

    # Determine the number of occurrence (consistency matrix). E.g. 
    # Gate | Splitter=1 | Splitter=0
    # G=1  | counts     | counts
    # G=0  | counts     | counts
    for testvalue in range(2):
        for splitvalue in range(2):
            df_temp = df_binary.loc[(df_binary[test_attribute] == testvalue) & (df_binary[parent_node]    == splitvalue)]
            count = df_temp.shape[0]
            c[(1 - testvalue), (1 - splitvalue)] = count

    return c


# ---------------------------------------------------------------------------
# Mantel-Haenszel Partial Association (PAMH) statistic
# ---------------------------------------------------------------------------

def pamh(counts):
    # Compute the PAMH statistic over a list of 2x2 contingency tables (strata).
    # A high score indicates a strong association between the gate and the parent.
    sumnumerator = 0.
    denominator = 0.
    
    for stratum in counts:
        if np.sum(stratum[:,0])==0 or np.sum(stratum[:,1])==0 or np.sum(stratum[0,:]) == 0 or np.sum(stratum[1,:])==0:
            continue
        else:
            above = stratum[0,0]*stratum[1,1] - stratum[1,0]*stratum[0,1]
            below = np.sum(stratum)
            stratumvalue = above / float(below)
            sumnumerator += stratumvalue
            
            #calculate denominator
            n1k = np.sum(stratum[0,:])
            n2k = np.sum(stratum[1,:])
            n1krow = np.sum(stratum[:,0])
            n2krow = np.sum(stratum[:,1])
            above = (n1k*n2k*n1krow*n2krow)
            total = np.sum(stratum)
            below = (total**2)*(total-1)
            value = above/float(below)
            denominator += value    
    sumnumerator = abs(sumnumerator)
    numerator = (sumnumerator - 0.5)**2
    if denominator == 0.:
        return 0
    else:
        return (numerator/float(denominator))


# ---------------------------------------------------------------------------
# Dominance threshold: controls how strict the pre-filter is before running the PAMH test.  
# We start strict (0.05) and relax up to (1 - significance) if no relationship is found at the current threshold.
# This ensures that stronger relationships are found first, even if CL is lower (no greedy search anymore).
# ---------------------------------------------------------------------------

_DOMINANCE_MIN  = 0.05  # Strictest threshold tried first, when significance is > 0.05 or CL < 0.95
_DOMINANCE_STEP = 0.01  # Increment when relaxing toward (1 - significance)

# When multiple gate types pass the PAMH test with the same score, prefer the more specific temporal gate.  
# This matters when the data is perfectly clean (top-down simulation): OR and SEQ produce identical contingency tables, 
# so without this tie-breaker OR always wins because it is tested first.
_GATE_SPECIFICITY = {'SEQ': 3, 'PAND': 3, 'AND': 2, 'OR': 1}


def testORgate(significance, df_itet, parent, tuplechildren, tolerance, dominance_threshold=None):
    # Test whether an OR gate with the given children significantly explains the parent.
    # The dominance pre-filter rejects the gate if too many rows contradict the expected direction
    # (avoids running the PAMH test on clearly wrong relationships).
    dt = dominance_threshold if dominance_threshold is not None else (1.0 - significance)

    dfextended = mergeOR(df_itet, tuplechildren, parent, tolerance)
    stratum    = getstratum(dfextended, parent, 'OR', [])
    righttop   = np.sum(stratum[0, 1])   # gate fires but parent does not
    leftbottom = np.sum(stratum[1, 0])   # parent fires but gate does not
    total      = np.sum(stratum)

    if righttop > dt * total:
        return False, 0.0
    if leftbottom > dt * total:
        return False, 0.0

    pamhscore = pamh([stratum])
    result = pamhscore >= getChi2(significance)
    return result, pamhscore


def testANDgate(significance, df_itet, parent, tuplechildren, tolerance, dominance_threshold=None):
    # Same structure as testORgate but for AND semantics.
    dt = dominance_threshold if dominance_threshold is not None else (1.0 - significance)

    dfextended = mergeAND(df_itet, tuplechildren, tolerance)
    stratum    = getstratum(dfextended, parent, 'AND', [])
    righttop   = np.sum(stratum[0, 1])
    leftbottom = np.sum(stratum[1, 0])
    total      = np.sum(stratum)

    if righttop > dt * total:
        return False, 0.0
    if leftbottom > dt * total:
        return False, 0.0

    pamhscore = pamh([stratum])
    result = pamhscore >= getChi2(significance)
    return result, pamhscore


def testPANDgate(significance, df, parent, tuplechildren, tolerance, dominance_threshold=None):
    # Test all permutations of the children to find the best causal ordering.
    # Returns the permutation with the highest PAMH score (if any passes).
    dt = dominance_threshold if dominance_threshold is not None else (1.0 - significance)

    result = False
    score = 0.0
    best_permutation = None

    for permuted_order in itertools.permutations(tuplechildren):
        dfextended = mergePAND(df, permuted_order, tolerance, parent=parent)
        stratum = getstratum(dfextended, parent, 'PAND', [])
        righttop = np.sum(stratum[0, 1])
        leftbottom = np.sum(stratum[1, 0])
        total = np.sum(stratum)

        if righttop > dt * total:
            continue
        if leftbottom > dt * total:
            continue

        pamhscore = pamh([stratum])
        if pamhscore >= getChi2(significance):
            result = True
            if pamhscore > score:
                score = pamhscore
                best_permutation = permuted_order

    return result, score, best_permutation


def testSEQvsPAND(df, best_permutation, significance, tolerance):
    # Decide whether a confirmed PAND relationship is actually SEQ (strict) or PAND (relaxed).
    # SEQ requires that all interval combinations are ordered (no violation allowed).
    # If violations occur at a statistically significant rate, we classify it as PAND.
    dfextended = mergeSEQ(df, best_permutation, tolerance)
    valid_seq_count    = np.sum(dfextended['SEQ'])
    interruption_count = np.sum(dfextended['SEQ_VIOLATION'])

    if VERBOSE:
        print('SEQ valid:', valid_seq_count)
        print('SEQ violation:', interruption_count)

    total = valid_seq_count + interruption_count
    significance_level = 1 - significance
    expected_interruptions = total * significance_level

    observed = [valid_seq_count, interruption_count]
    expected = [total - expected_interruptions, expected_interruptions]

    chi2_stat, p_value = chisquare(observed, f_exp=expected)

    # If the interruption rate is statistically significant, it is PAND (not SEQ)
    if p_value < significance_level:
        return 'PAND'
    else:
        return 'SEQ'


# ---------------------------------------------------------------------------
# Core layer learning
# ---------------------------------------------------------------------------

def createlayer(significance, df_itet, generatedtree, seen_parent, parentlist, tolerance):
    # For each parent node in parentlist, find the best gate relationship that explains it from the available child columns.
    #
    # Search strategy:
    #   1. Try the smallest subset size first (prefer simpler relationships).
    #   2. At each size, start with a strict dominance threshold (0.05) and gradually relax it until a relationship is found or the maximum is reached.
    #   3. Among all passing gates (OR, AND, PAND/SEQ), keep the one with the highest PAMH score.
    #   4. Stop as soon as a size yields at least one candidate.

    df_bool = df_itet.map(to_bool)

    for parent_node in parentlist:
        if VERBOSE:
            print(f"Processing parent_node: {parent_node}")

        df_static = df_itet.copy()

        # Drop events that are never active in rows where the parent fires, because they cannot explain the parent.
        parent_active_mask = df_bool[parent_node] == 1
        dead_events = []
        for col in df_static.columns:
            if col == parent_node or col in seen_parent:
                continue
            if (df_bool.loc[parent_active_mask, col] == 0).all():
                dead_events.append(col)

        df_static = df_static.drop(columns=dead_events)
        if VERBOSE and dead_events:
            print(f"  Excluded (never active when {parent_node}=1): {dead_events}")

        availablesets = getavailablesets(df_static, parent_node, seen_parent)
        candidates    = []

        if availablesets:
            max_dominance = 1.0 - significance

            # Group subsets by their size so we can iterate smallest-first.
            sets_by_size = {}
            for a in availablesets:
                size = len(a)
                if size not in sets_by_size:
                    sets_by_size[size] = []
                sets_by_size[size].append(a)

            for size in sorted(sets_by_size.keys()):
                subsets = sets_by_size[size]

                # Start with the strictest dominance threshold and relax if needed.
                dt = min(_DOMINANCE_MIN, max_dominance)

                while True:
                    best_OR = None
                    best_AND = None
                    best_PAND = None

                    for a in subsets:
                        # --- OR ---
                        ok, score = testORgate(significance, df_static, parent_node, a, tolerance, dt)
                        if ok:
                            if best_OR is None or score > best_OR[0]:
                                best_OR = (score, a, 'OR')

                        # --- AND ---
                        ok, score = testANDgate(significance, df_static, parent_node, a, tolerance, dt)
                        if ok:
                            if best_AND is None or score > best_AND[0]:
                                best_AND = (score, a, 'AND')

                        # --- PAND / SEQ ---
                        ok, score, best_perm = testPANDgate(significance, df_static, parent_node, a, tolerance, dt)
                        if ok:
                            if VERBOSE:
                                print(f'  Best permutation for PAND: {best_perm} (parent: {parent_node})')
                            if best_PAND is None or score > best_PAND[0]:
                                gate = testSEQvsPAND(df_static, best_perm, significance, tolerance)
                                best_PAND = (score, best_perm, gate)

                    # Collect all gate types that passed.
                    candidates = [c for c in (best_OR, best_AND, best_PAND) if c is not None]

                    if candidates:
                        # Found something at this dominance threshold — stop relaxing.
                        if VERBOSE:
                            print(f"  Size={size}, dominance threshold={dt:.3f}")
                        break

                    if dt >= max_dominance - 1e-9:
                        # Already at maximum allowed threshold — nothing found at this size
                        break

                    # Relax the threshold by one step and try again
                    dt = min(round(dt + _DOMINANCE_STEP, 4), max_dominance)

                if candidates:
                    break  # Found a relationship at this size — do not try larger subsets

        if candidates:
            # Among all passing gate types, pick the highest PAMH score.
            # Break ties in favor of more specific gate types (SEQ/PAND > AND > OR).
            best = max(candidates, key=lambda c: (c[0], _GATE_SPECIFICITY.get(c[2], 0)))
            pamhscore_best, children_best, gate_best = best
            generatedtree[parent_node] = [children_best, gate_best]

        if parent_node in generatedtree:
            seen_parent.append(parent_node)

    return generatedtree, seen_parent


# ---------------------------------------------------------------------------
# Unused BE cleanup
# ---------------------------------------------------------------------------

def remove_unused_basic_events(tree):
    # Remove any BE node that is not referenced as a child by any gate node. This keeps the tree clean after each layer is built.
    all_keys   = set(tree.keys())
    referenced = set()

    for node, (children, gate) in tree.items():
        if gate == 'BE':
            continue
        for child in children:
            if isinstance(child, tuple):
                referenced.update(child)
            else:
                referenced.add(child)

    unused_bes = []
    for node in all_keys:
        if tree[node][1] == 'BE' and node not in referenced:
            unused_bes.append(node)

    for node in unused_bes:
        del tree[node]

    return tree, unused_bes


# ---------------------------------------------------------------------------
# Full depth-N DFT learning
# ---------------------------------------------------------------------------

def learnFTandcheck(tree, df_itet, significance, top_event="TE", interval_tolerance=INTERVAL_TOLERANCE):
    """
    Learn a full DFT layer by layer until no new relationships are found.
    Returns (match, learned_tree) where match indicates whether the learned tree matches the provided reference tree.
    """
    # Drop columns that never fire, because they cannot contribute to any gate.
    cols_to_drop = [col for col in df_itet.columns if df_itet[col].map(to_bool).sum() == 0]
    df_itet = df_itet.drop(columns=cols_to_drop)
    df_itet = df_itet.copy()

    newtree = tree.copy()
    generatedtree = dict()
    seen_parent = [top_event]
    oldseen_parent = []

    # First layer: learn the gate for the TE
    generatedtree, seen_parent = createlayer(significance, df_itet, generatedtree, seen_parent, [top_event], interval_tolerance)

    # Continue layer by layer until the seen_parent set stops growing.
    while seen_parent != oldseen_parent:
        # Collect all children from the current tree that are not yet gate nodes.
        parentlist = []
        for node, (children, gate) in generatedtree.items():
            for child in children:
                if child not in generatedtree:
                    parentlist.append(child)

        oldseen_parent = copy.deepcopy(seen_parent)

        generatedtree, seen_parent = createlayer(significance, df_itet, generatedtree, seen_parent, parentlist, interval_tolerance)
        generatedtree, _ = remove_unused_basic_events(generatedtree)

        if VERBOSE:
            print('temp tree: ', generatedtree)

    # Mark all remaining columns as BEs.
    for event in df_itet.columns:
        if event not in generatedtree and event not in seen_parent:
            generatedtree[event] = [(), 'BE']

    generatedtree, _ = remove_unused_basic_events(generatedtree)

    # Compare with the reference tree and report any differences
    if newtree != generatedtree:
        for key in tree:
            if newtree[key] != generatedtree.get(key):
                if VERBOSE:
                    print(f"Difference found for key '{key}':")
                    print("  reference tree :", tree[key])
                    print("  generated tree :", generatedtree.get(key))
        return False, generatedtree
    else:
        return True, generatedtree


def learn_depthN_DFT_with_most_significant_relationship(df_itet, significance, n_depth=1, top_event="TE", interval_tolerance=INTERVAL_TOLERANCE):
    """
    Learn a DFT up to a fixed depth n_depth. At each depth level, only the single best gate is added per parent node.
    """

    # Drop columns that never fire, because they cannot contribute to any gate.
    cols_to_drop = [col for col in df_itet.columns if df_itet[col].map(to_bool).sum() == 0]
    df_itet = df_itet.drop(columns=cols_to_drop)

    generatedtree = {}
    seen_parent = [top_event]
    current_parent_nodes = [top_event]

    for _ in range(n_depth):
        if not current_parent_nodes:
            break

        generatedtree, seen_parent = createlayer(significance, df_itet, generatedtree, seen_parent, current_parent_nodes, interval_tolerance)

        # Collect the children of this layer as the next layer's parent nodes
        next_parent_nodes = []
        for parent_node in current_parent_nodes:
            if parent_node not in generatedtree:
                continue
            for child in generatedtree[parent_node][0]:
                if child not in generatedtree and child not in next_parent_nodes:
                    next_parent_nodes.append(child)

        current_parent_nodes = next_parent_nodes

    # All remaining columns become BEs
    for event in df_itet.columns:
        if event not in generatedtree:
            generatedtree[event] = [(), 'BE']

    generatedtree, _ = remove_unused_basic_events(generatedtree)

    return generatedtree


# ---------------------------------------------------------------------------
# Depth-1 learning with ranked relationships
# ---------------------------------------------------------------------------

def _collect_relations_with_scores(significance, df_itet, parent_node, seen_parent, tolerance):
    # Find all significant gate relationships for parent_node and return them sorted by PAMH score (highest first).
    # Uses the same dominance threshold relaxation as createlayer: start strict, relax if nothing found.
    # Uses testSEQvsPAND to correctly distinguish SEQ from PAND.
    # Deduplicates by (gate, child set).
    best = {}  # key: (gate, frozenset(children))  -->  value: (score, ordered_children)

    max_dominance = 1.0 - significance
    dt = min(_DOMINANCE_MIN, max_dominance)

    while True:
        for a in getavailablesets(df_itet, parent_node, seen_parent):
            # OR
            ok, score = testORgate(significance, df_itet, parent_node, a, tolerance, dt)
            if ok:
                key = ("OR", frozenset(a))
                if score > best.get(key, (0,))[0]:
                    best[key] = (score, tuple(a))

            # AND
            ok, score = testANDgate(significance, df_itet, parent_node, a, tolerance, dt)
            if ok:
                key = ("AND", frozenset(a))
                if score > best.get(key, (0,))[0]:
                    best[key] = (score, tuple(a))

            # PAND / SEQ
            ok, score, best_perm = testPANDgate(significance, df_itet, parent_node, a, tolerance, dt)
            if ok and best_perm is not None:
                gate = testSEQvsPAND(df_itet, best_perm, significance, tolerance)
                key = (gate, frozenset(best_perm))
                if score > best.get(key, (0,))[0]:
                    best[key] = (score, tuple(best_perm))

        if best:
            break
        if dt >= max_dominance - 1e-9:
            break
        dt = min(round(dt + _DOMINANCE_STEP, 4), max_dominance)

    # Build the final list and sort by score descending, breaking ties by gate specificity.
    relations = []
    for (gate, _), (score, children) in best.items():
        relations.append((score, gate, children))

    relations.sort(key=lambda x: (x[0], _GATE_SPECIFICITY.get(x[1], 0)), reverse=True)
    return relations


def learn_depth1_DFT_with_N_significant_relationships(df_itet, significance, N=2, top_event="TE", interval_tolerance=INTERVAL_TOLERANCE):
    """
    Learn a depth-1 DFT keeping the N most significant relationships to the TE, ranked by PAMH score.

    N=0: keep all significant relationships
    N=1: TE gets the single best gate directly
    N>1: TE gets OR(rel_1, ..., rel_N)
    """

    cols_to_drop = [col for col in df_itet.columns if df_itet[col].map(to_bool).sum() == 0]
    df_itet = df_itet.drop(columns=cols_to_drop)

    all_relations = _collect_relations_with_scores(
        significance=significance,
        df_itet=df_itet,
        parent_node=top_event,
        seen_parent=[top_event],
        tolerance=interval_tolerance,
    )

    # N=0 means keep everything; otherwise take the top N
    if N == 0:
        top_relations = all_relations
    else:
        top_relations = all_relations[:N]

    depth1_tree = {}

    # No relationships found — return a tree with only BEs
    if not top_relations:
        for event in df_itet.columns:
            depth1_tree[event] = [(), "BE"]
        depth1_tree, _ = remove_unused_basic_events(depth1_tree)
        return depth1_tree

    if len(top_relations) == 1:
        # Single relationship — connect it directly to the TE
        _, gate, children = top_relations[0]
        depth1_tree[top_event] = [tuple(children), gate]
    else:
        # Multiple relationships — combine them under an OR gate
        rel_nodes = []
        for i, (_, gate, children) in enumerate(top_relations, start=1):
            rel_name = f"{top_event}_rel_{i}"
            depth1_tree[rel_name] = [tuple(children), gate]
            rel_nodes.append(rel_name)
        depth1_tree[top_event] = [tuple(rel_nodes), "OR"]

    # Register all remaining columns as BEs
    for event in df_itet.columns:
        if event not in depth1_tree:
            depth1_tree[event] = [(), "BE"]

    depth1_tree, _ = remove_unused_basic_events(depth1_tree)

    # Put the TE first in the dict for readability
    ordered = {top_event: depth1_tree[top_event]}
    for k, v in depth1_tree.items():
        if k != top_event:
            ordered[k] = v

    return ordered


# ---------------------------------------------------------------------------
# ITET Simulation
# ---------------------------------------------------------------------------

def _union_merge(result, updates):
    """
    Write every entry from updates into result, with one special rule for shared events:
    If both result and updates contain intervals (frozensets) for the same node, combine them into one frozenset that contains all intervals from both sides.
    If one side has intervals and the other has 0 ("did not fire"), keep the intervals.
    This matters when a BE appears under more than one gate. Each gate branch writes its own value for that event, 
    and a "did not fire" decision in one branch must not erase intervals that were legitimately generated by another branch.
    
    Example: an event is shared between a PAND branch and an AND branch.
      result['pump_fail']  = frozenset([(1.0, 2.5)])  # PAND branch recorded t=1.0..2.5
      updates['pump_fail'] = frozenset([(4.0, 5.1)])  # AND branch recorded t=4.0..5.1
      --> result['pump_fail'] = frozenset([(1.0, 2.5), (4.0, 5.1)])  # both occurrences preserved
    """
    for k, v in updates.items():
        if k not in result:
            result[k] = v
        elif isinstance(result[k], frozenset) and isinstance(v, frozenset):
            result[k] = result[k] | v
        elif not isinstance(result[k], frozenset):
            result[k] = v


def _gen_ordered_intervals(n, range_seconds, min_duration, tolerance):
    """
    Generate n intervals that satisfy the PAND ordering conditions by construction (conditions are automatically satisfied).

    The two conditions that must hold for each consecutive pair (i, i+1):
      starts[i] + tolerance < starts[i+1] --> i+1 starts strictly after i (with a safety gap)
      starts[i+1] < ends[i] --> i is still active when i+1 starts (co-active)

    The key idea: instead of generating intervals randomly and checking afterwards,
    we draw each start s_i directly from INSIDE the previous interval (prev_s + step, prev_e).
    This placement guarantees both conditions above by construction — no rejection needed.

    step = tolerance + 0.01 ensures the strict inequality (> tolerance, not >=).
    min_duration = 0.2 ensures the available window is always large enough to fit the step.
    """
    # Minimum gap between consecutive start times: must exceed tolerance for starts_before_and_overlaps.
    # The +0.01 ensures the strict inequality a[0] + tolerance < b[0] is always satisfied after rounding.
    step = tolerance + 0.01

    if n == 1:
        s = round(random.uniform(0, max(0, range_seconds - min_duration)), 3)
        e = round(random.uniform(s + min_duration, min(s + range_seconds * 0.5, range_seconds)), 3)
        return [(s, e)]

    intervals = []

    # First interval: its end must extend far enough past s0+step so that s1 can be drawn inside it.
    # max_s0 leaves enough room for all n intervals to fit within range_seconds.
    max_s0 = max(0.0, range_seconds - n * (step + min_duration))
    s0 = round(random.uniform(0, max_s0), 3)
    min_e0 = s0 + step + 0.001   # guarantees the range (s0+step, e0) is non-empty for s1
    max_e0 = min(s0 + range_seconds * 0.7, range_seconds)
    if min_e0 > max_e0:
        max_e0 = min_e0  # fallback: accept minimum end even if it exceeds the soft cap
    e0 = round(random.uniform(min_e0, max_e0), 3)
    intervals.append((s0, e0))

    for i in range(1, n):
        prev_s, prev_e = intervals[-1]

        # Draw s_i from strictly inside (prev_s + step, prev_e - epsilon).
        # Being inside the previous interval simultaneously satisfies both PAND conditions:
        #   s_i > prev_s + step --> ordered starts (condition 1)
        #   s_i < prev_e --> prev interval is still active (condition 2)
        lo = prev_s + step
        hi = prev_e - 0.001
        if lo >= hi:
            hi = lo + 0.001  # fallback for very tight windows (window < step)
        si = round(random.uniform(lo, hi), 3)

        if i < n - 1:
            # Not the last interval: end must leave room for the next start inside it.
            min_ei = si + step + 0.001
        else:
            # Last interval: only needs to meet the minimum duration.
            min_ei = si + min_duration
        max_ei = min(si + range_seconds * 0.6, range_seconds)
        if min_ei > max_ei:
            max_ei = min_ei
        ei = round(random.uniform(min_ei, max_ei), 3)
        intervals.append((si, ei))

    return intervals


def _sample_row_topdown(dft, top_event, should_fire, range_seconds,
                        p_or_child_fire, min_duration, tolerance):
    """
    Generate one row of the ITET by propagating the firing decision top-down through the DFT.

    TOP DOWN:
      A naive bottom-up approach would generate BE intervals independently and then check whether the gate conditions happen to be satisfied by chance. 
      This works poorly for strict gates (PAND, SEQ, AND) where the chance of accidental satisfaction is low. 
      Instead, we first decide for each node whether it fires (top-down), and then generate intervals that are GUARANTEED to satisfy the gate semantics.

    forced_iv:
      When a parent gate requires a child to fire at a specific time, so that the parent's own interval is consistent.
      It passes forced_iv down to the child. The child then uses that interval as its own output interval and generates its subtree inside it.
      Without forced_iv, a PAND child would generate an independent interval that may not overlap with the parent's interval, causing false negatives.

    Returns a dict: node_name --> frozenset of (start, end) tuples (or 0 if did not fire),
    for every node in the DFT except the TE itself.
    """


    def _free_iv():
        # Generate a random interval anywhere in the observation window [0, range_seconds].
        s = round(random.uniform(0, max(0, range_seconds - min_duration)), 3)
        e = round(random.uniform(s + min_duration, min(s + range_seconds * 0.5, range_seconds)), 3)
        return (s, e)


    def generate(node, fires, forced_iv=None):
        children, gate = dft[node]

        # --- Basic Event ---
        # A BE has no children. If it fires, it gets one interval (forced or random).
        if gate == 'BE':
            if fires:
                iv = forced_iv if forced_iv else _free_iv()
                return {node: frozenset([iv])}
            else:
                return {node: 0}

        # --- AND gate ---
        if gate == 'AND':
            if fires:
                # All children must be simultaneously active (non-empty intersection).
                # Giving all children the SAME interval guarantees the intersection is non-empty.
                iv     = forced_iv if forced_iv else _free_iv()
                result = {node: frozenset([iv])}
                for child in children:
                    _union_merge(result, generate(child, True, iv))
                return result
            else:
                # All children forced to 0. If only one were forced off (and the others fired randomly),
                # a shared BE could receive intervals deposited by a sibling gate that IS firing and accidentally satisfy this AND condition.
                result = {node: 0}
                for child in children:
                    _union_merge(result, generate(child, False))
                return result

        # --- OR gate ---
        if gate == 'OR':
            if fires:
                # Exactly one child is forced to fire with the gate's interval (guarantees OR fires).
                # The remaining children fire independently with probability p_or_child_fire.
                # The forced child receives the gate's interval so the parent's overlap check succeeds.
                iv          = forced_iv if forced_iv else _free_iv()
                forced_idx  = random.randint(0, len(children) - 1)
                result      = {node: frozenset([iv])}
                for i, child in enumerate(children):
                    if i == forced_idx:
                        _union_merge(result, generate(child, True, iv))
                    else:
                        child_fires = random.random() < p_or_child_fire
                        _union_merge(result, generate(child, child_fires))
                return result
            else:
                # OR does not fire — no child is allowed to fire.
                result = {node: 0}
                for child in children:
                    _union_merge(result, generate(child, False))
                return result

        # --- PAND / SEQ gate ---
        if gate in ('PAND', 'SEQ'):
            if fires:
                if forced_iv is not None:
                    # A parent passed a specific window down to this gate.
                    # Generate the ordered child intervals inside that window, then assign the window itself as the gate's top-down output interval (temporary).
                    #  _evaluate_bottomup later replaces this with the last child's actual interval (ivs[-1]).
                    # The no-false-negatives property is preserved because ivs[-1] lies inside the forced window, 
                    # so the gate remains non-zero after bottom-up recomputation.
                    # 'available' shrinks the window by tolerance + epsilon so that every child start
                    # satisfies  child_start + tolerance < w_e  after shifting — required for intervals_overlap.
                    w_s, w_e = forced_iv
                    available = max(0.001, w_e - w_s - tolerance - 0.001)
                    ivs = _gen_ordered_intervals(len(children), available, min_duration, tolerance)
                    ivs = [(round(s + w_s, 3), round(e + w_s, 3)) for s, e in ivs]
                    gate_iv = forced_iv
                else:
                    # No forced window — generate freely and use the last child's interval as the gate output.
                    # The last child determines when the ordered sequence completes, which is the natural output time of a PAND/SEQ gate.
                    ivs = _gen_ordered_intervals(len(children), range_seconds, min_duration, tolerance)
                    gate_iv = ivs[-1]
                result = {node: frozenset([gate_iv])}
                for i, child in enumerate(children):
                    _union_merge(result, generate(child, True, ivs[i]))
                return result
            else:
                # Two strategies to make PAND/SEQ not fire, chosen to produce realistic negative rows:
                #
                # Strategy 1 — first child absent (used for both PAND and SEQ):
                #   The causal chain is broken because the triggering event never occurred. Remaining children fire randomly.
                #
                # Strategy 2 — reversed order (used for PAND only, 50% of not-fire rows):
                #   All children fire, but in the wrong order (last child first).
                #   This is a genuine PAND violation — events occurred but out of sequence.
                #   SEQ never uses this strategy: SEQ data must have zero ordering violations,
                #   so testSEQvsPAND can correctly classify it as SEQ rather than PAND.
                use_reversed = (gate == 'PAND') and (random.random() < 0.5)
                if not use_reversed:
                    result = {node: 0}
                    _union_merge(result, generate(children[0], False))
                    for child in children[1:]:
                        child_fires = random.random() < 0.5
                        _union_merge(result, generate(child, child_fires))
                    return result
                else:
                    # Assign intervals in reverse order: last child gets the earliest interval.
                    ivs = _gen_ordered_intervals(len(children), range_seconds, min_duration, tolerance)
                    result = {node: 0}
                    for i, child in enumerate(children):
                        _union_merge(result, generate(child, True, ivs[len(children) - 1 - i]))
                    return result

        return {}

    return generate(top_event, should_fire)


def _topological_order(dft, top_event):
    # Post-order DFS — leaves first, TE last.
    order = []
    visited = set()

    def visit(node):
        if node in visited or node not in dft:
            return
        visited.add(node)
        for child in dft[node][0]:
            visit(child)
        order.append(node)

    visit(top_event)
    return order


def _evaluate_bottomup(dft, top_event, node_values, tolerance, nodes_to_recompute=None):
    # Recompute IE intervals by applying gate semantics to the current child intervals in node_values.
    # If nodes_to_recompute is given (a set), only those nodes are updated. All others keep their top-down values unchanged.
    result = dict(node_values)

    for node in _topological_order(dft, top_event):
        if nodes_to_recompute is not None and node not in nodes_to_recompute:
            continue

        children, gate = dft[node]
        if not children:  # BE — keep actual intervals unchanged
            continue

        child_ivs = [get_intervals(result.get(c, 0)) for c in children]

        if gate == 'OR':
            all_ivs = [iv for ivs in child_ivs for iv in ivs]
            result[node] = frozenset(all_ivs) if all_ivs else 0

        elif gate == 'AND':
            if any(len(ivs) == 0 for ivs in child_ivs):
                result[node] = 0
                continue
            and_iv = None
            for combo in itertools.product(*child_ivs):
                sect = combo[0]
                for iv in combo[1:]:
                    sect = interval_intersection(sect, iv, tolerance)
                    if sect is None:
                        break
                if sect is not None:
                    and_iv = sect
                    break
            result[node] = frozenset([and_iv]) if and_iv is not None else 0

        elif gate in ('PAND', 'SEQ'):
            if any(len(ivs) == 0 for ivs in child_ivs):
                result[node] = 0
                continue
            pand_iv = None
            for combo in itertools.product(*child_ivs):
                if all(
                    starts_before_and_overlaps(combo[j], combo[j + 1], tolerance)
                    for j in range(len(combo) - 1)
                ):
                    pand_iv = combo[-1]
                    break
            result[node] = frozenset([pand_iv]) if pand_iv is not None else 0

    return result


def _find_shared_nodes(dft):
    """Return the frozenset of nodes (BE or IE) that appear as a child of more than one gate."""
    child_parent_count = {}
    for node, (children, gate) in dft.items():
        if gate == 'BE':
            continue
        for child in children:
            child_parent_count[child] = child_parent_count.get(child, 0) + 1
    return frozenset(node for node, n in child_parent_count.items() if n > 1)


def _ancestors_of(dft, nodes):
    """
    Return the set of all non-BE nodes that are ancestors of any node in 'nodes' (direct parents and transitively upward). 
    These are the IEs whose values depend on the given nodes and must be recomputed bottom-up.
    """
    parent_map = {}
    for node, (children, _) in dft.items():
        for child in children:
            parent_map.setdefault(child, set()).add(node)

    affected = set()
    queue = list(nodes)
    visited = set(nodes)

    while queue:
        current = queue.pop()
        for parent in parent_map.get(current, set()):
            if parent not in visited:
                visited.add(parent)
                if dft[parent][1] != 'BE':
                    affected.add(parent)
                    queue.append(parent)

    return affected


def ITET_simulation(dft, range_seconds, n_samples, top_event=None, p_TE=0.5, p_or_child_fire=0.5, min_duration=0.2, interval_tolerance=INTERVAL_TOLERANCE, random_seed=None):
    """
    Generate a synthetic ITET DataFrame from a DFT using top-down sampling followed by bottom-up recomputation (if shared events available).

    For each row the simulation first decides whether the TE fires (with probability p_TE),
    then generates child intervals that are guaranteed to produce exactly that outcome.
    Gate semantics are enforced by construction, not by chance.
    For DFTs with shared nodes, the IE ancestors of the shared nodes are subsequently recomputed bottom-up 
    by applying gate semantics to the current child intervals, ensuring consistency for nodes affected by the shared topology.
    Pure trees skip this step entirely, as the top-down simulation already produces consistent values.

    Gate behaviour (firing):
      AND: all children receive the same interval — intersection is always non-empty
      OR: one child is forced to fire; others fire independently with probability p_or_child_fire
      PAND/SEQ: children receive ordered intervals, each start drawn inside the previous interval

    Gate behaviour (not firing):
      AND: all children are forced to 0
      OR:   all children are forced to 0
      PAND: either the first child is absent, or all children fire in reversed order (50/50)
      SEQ:  first child is absent; remaining children fire independently

    Parameters
    ----------
    dft: reference DFT dict that defines the structure to simulate
    range_seconds: length of the observation window in seconds
    n_samples: number of rows to generate
    top_event: name of the TE (auto-detected if None)
    p_TE: fraction of rows where the TE fires (label balance)
    p_or_child_fire: probability that a non-forced OR child fires
    min_duration: minimum interval duration in seconds
    interval_tolerance: tolerance for gate evaluation (same value used in learning)
    random_seed: integer seed for reproducibility

    Returns
    -------
    pd.DataFrame with one column per node (frozenset or 0) and one binary label column for the TE (0 or 1).
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # Auto-detect the TE: the node that is not a child of any other node
    if top_event is None:
        all_children = set()
        for _, (children, _) in dft.items():
            for c in children:
                all_children.add(c)

        candidates = [n for n in dft if n not in all_children]
        if not candidates:
            raise ValueError("Could not auto-detect TE.")
        top_event = candidates[0]

    # All nodes except the TE become columns in the dataset.
    all_nodes = [node for node in dft if node != top_event]

    # Detect shared events once. For pure trees (no shared events) the top-down simulation already produces consistent values.
    # So bottom-up recomputation is skipped entirely (it can corrupt carefully-constructed PAND/SEQ orderings).
    # When shared events exist, only the ancestors of those events need recomputation.
    shared_nodes   = _find_shared_nodes(dft)
    affected_nodes = _ancestors_of(dft, shared_nodes) if shared_nodes else None

    rows = []
    for _ in range(n_samples):
        should_fire = random.random() < p_TE

        node_values = _sample_row_topdown(dft, top_event, should_fire, range_seconds, p_or_child_fire, min_duration, interval_tolerance)

        if affected_nodes:
            node_values = _evaluate_bottomup(dft, top_event, node_values, interval_tolerance, affected_nodes)

        row = {}
        for node in all_nodes:
            row[node] = node_values.get(node, 0)
        row[top_event] = 1 if should_fire else 0
        rows.append(row)

    cols = all_nodes + [top_event]
    return pd.DataFrame(rows, columns=cols)
