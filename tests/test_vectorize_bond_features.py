"""Test vectorized bond features computation.

The original implementation used O(N_token^2) nested Python loops to check
atom-level bonds between every token pair. For a 1000-token target, that's
~500K iterations in pure Python.

The vectorized version uses NumPy advanced indexing on the atom adjacency
matrix, mapping bonded atom pairs directly to their token indices. This
eliminates the inner loops entirely.

Expected: ~50-100x speedup on the bond feature computation.
"""

import time

import numpy as np
import torch


def _make_mock_data(num_tokens, atoms_per_token, num_bonds):
    """Create mock data that exercises the bond feature logic."""
    num_atoms = num_tokens * atoms_per_token

    # Create mock token array
    class MockToken:
        def __init__(self, atom_indices):
            self.atom_indices = atom_indices

    class MockTokenArray:
        def __init__(self, tokens):
            self.tokens = tokens

        def __len__(self):
            return len(self.tokens)

        def __getitem__(self, idx):
            return self.tokens[idx]

    tokens = []
    for i in range(num_tokens):
        start = i * atoms_per_token
        atom_indices = list(range(start, start + atoms_per_token))
        tokens.append(MockToken(atom_indices))
    token_array = MockTokenArray(tokens)

    # Create mock atom array with annotations
    mol_types = np.array(["protein"] * num_atoms)
    res_names = np.array(["ALA"] * num_atoms)  # Standard residue
    ref_space_uids = np.array([i // atoms_per_token for i in range(num_atoms)])

    # Make some tokens ligands
    n_ligand_tokens = max(1, num_tokens // 10)
    for i in range(num_tokens - n_ligand_tokens, num_tokens):
        start = i * atoms_per_token
        mol_types[start:start + atoms_per_token] = "ligand"
        res_names[start:start + atoms_per_token] = "LIG"

    # Create bonds (sparse adjacency)
    class MockBonds:
        def __init__(self, n_atoms, n_bonds):
            # Create random bonds, ensuring some cross-token bonds
            rows, cols = [], []
            for _ in range(n_bonds):
                a = np.random.randint(0, n_atoms)
                b = np.random.randint(0, n_atoms)
                if a != b:
                    rows.extend([a, b])
                    cols.extend([b, a])
            self._rows = np.array(rows)
            self._cols = np.array(cols)
            self._n = n_atoms

        def adjacency_matrix(self):
            mat = np.zeros((self._n, self._n), dtype=int)
            if len(self._rows) > 0:
                mat[self._rows, self._cols] = 1
            return mat

    class MockAtomArray:
        def __init__(self):
            self.mol_type = mol_types
            self.res_name = res_names
            self.ref_space_uid = ref_space_uids
            self.bonds = MockBonds(num_atoms, num_bonds)

        def __len__(self):
            return num_atoms

    return token_array, MockAtomArray()


def _bond_features_original(token_array, atom_array):
    """Original O(N^2) implementation for reference."""
        # Inline constants to avoid importing pxdesign.data.constants (requires rdkit)
    DESIGN_RESIDUES = {"xpb": 32, "xpa": 33, "rbb": 34, "raa": 35}
    STD_RESIDUES = {
        "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4, "GLN": 5, "GLU": 6,
        "GLY": 7, "HIS": 8, "ILE": 9, "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13,
        "PRO": 14, "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19, "UNK": 20,
        "A": 21, "G": 22, "C": 23, "U": 24, "N": 25,
        "DA": 26, "DG": 27, "DC": 28, "DT": 29, "DN": 30,
        "xpb": 32, "xpa": 33, "rbb": 34, "raa": 35,
    }

    num_tokens = len(token_array)
    adj_matrix = atom_array.bonds.adjacency_matrix().astype(int)
    token_adj_matrix = np.zeros((num_tokens, num_tokens), dtype=int)
    atom_bond_mask = adj_matrix > 0

    for i in range(num_tokens):
        atoms_i = token_array[i].atom_indices
        token_i_mol_type = atom_array.mol_type[atoms_i[0]]
        token_i_res_name = atom_array.res_name[atoms_i[0]]
        if token_i_res_name in DESIGN_RESIDUES:
            continue
        token_i_ref_space_uid = atom_array.ref_space_uid[atoms_i[0]]
        unstd_res_token_i = (
            token_i_res_name not in STD_RESIDUES and token_i_mol_type != "ligand"
        )
        is_polymer_i = token_i_mol_type in ["protein", "dna", "rna"]

        for j in range(i + 1, num_tokens):
            atoms_j = token_array[j].atom_indices
            token_j_mol_type = atom_array.mol_type[atoms_j[0]]
            token_j_res_name = atom_array.res_name[atoms_j[0]]
            token_j_ref_space_uid = atom_array.ref_space_uid[atoms_j[0]]
            unstd_res_token_j = (
                token_j_res_name not in STD_RESIDUES
                and token_j_mol_type != "ligand"
            )
            is_polymer_j = token_j_mol_type in ["protein", "dna", "rna"]

            if is_polymer_i and is_polymer_j:
                is_same_res = token_i_ref_space_uid == token_j_ref_space_uid
                unstd_res_bonds = unstd_res_token_i and unstd_res_token_j
                if not (is_same_res and unstd_res_bonds):
                    continue

            sub_matrix = atom_bond_mask[np.ix_(atoms_i, atoms_j)]
            if np.any(sub_matrix):
                token_adj_matrix[i, j] = 1
                token_adj_matrix[j, i] = 1

    return token_adj_matrix


def _bond_features_vectorized(token_array, atom_array):
    """Vectorized implementation."""
        # Inline constants to avoid importing pxdesign.data.constants (requires rdkit)
    DESIGN_RESIDUES = {"xpb": 32, "xpa": 33, "rbb": 34, "raa": 35}
    STD_RESIDUES = {
        "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4, "GLN": 5, "GLU": 6,
        "GLY": 7, "HIS": 8, "ILE": 9, "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13,
        "PRO": 14, "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19, "UNK": 20,
        "A": 21, "G": 22, "C": 23, "U": 24, "N": 25,
        "DA": 26, "DG": 27, "DC": 28, "DT": 29, "DN": 30,
        "xpb": 32, "xpa": 33, "rbb": 34, "raa": 35,
    }

    num_tokens = len(token_array)
    num_atoms = len(atom_array)

    atom_to_token = np.full(num_atoms, -1, dtype=np.int64)
    for idx, token in enumerate(token_array.tokens):
        for atom_idx in token.atom_indices:
            atom_to_token[atom_idx] = idx

    first_atoms = np.array([token_array[i].atom_indices[0] for i in range(num_tokens)])
    token_mol_type = atom_array.mol_type[first_atoms]
    token_res_name = atom_array.res_name[first_atoms]
    token_ref_space_uid = atom_array.ref_space_uid[first_atoms]

    is_design = np.array([rn in DESIGN_RESIDUES for rn in token_res_name])
    is_polymer = np.isin(token_mol_type, ["protein", "dna", "rna"])
    is_ligand = token_mol_type == "ligand"
    is_std = np.array([rn in STD_RESIDUES for rn in token_res_name])
    is_unstd = ~is_std & ~is_ligand

    adj_matrix = atom_array.bonds.adjacency_matrix()
    bonded_i, bonded_j = np.nonzero(adj_matrix)

    tok_i = atom_to_token[bonded_i]
    tok_j = atom_to_token[bonded_j]

    inter_token = tok_i != tok_j
    tok_i = tok_i[inter_token]
    tok_j = tok_j[inter_token]

    valid = ~is_design[tok_i] & ~is_design[tok_j]
    tok_i = tok_i[valid]
    tok_j = tok_j[valid]

    both_polymer = is_polymer[tok_i] & is_polymer[tok_j]
    same_res = token_ref_space_uid[tok_i] == token_ref_space_uid[tok_j]
    both_unstd = is_unstd[tok_i] & is_unstd[tok_j]
    polymer_allowed = same_res & both_unstd
    exclude = both_polymer & ~polymer_allowed
    tok_i = tok_i[~exclude]
    tok_j = tok_j[~exclude]

    token_adj_matrix = np.zeros((num_tokens, num_tokens), dtype=int)
    if len(tok_i) > 0:
        token_adj_matrix[tok_i, tok_j] = 1

    return token_adj_matrix


def test_vectorized_matches_original():
    """Verify vectorized output matches original for various configurations."""
    np.random.seed(42)
    for num_tokens, atoms_per_token, num_bonds in [
        (20, 5, 50),
        (50, 3, 100),
        (100, 4, 200),
    ]:
        token_array, atom_array = _make_mock_data(num_tokens, atoms_per_token, num_bonds)
        original = _bond_features_original(token_array, atom_array)
        vectorized = _bond_features_vectorized(token_array, atom_array)
        np.testing.assert_array_equal(
            original, vectorized,
            err_msg=f"Mismatch for {num_tokens} tokens, {atoms_per_token} atoms/token, {num_bonds} bonds"
        )


def test_vectorized_speedup():
    """Benchmark vectorized vs original implementation."""
    np.random.seed(42)
    num_tokens = 500
    atoms_per_token = 5
    num_bonds = 1000
    token_array, atom_array = _make_mock_data(num_tokens, atoms_per_token, num_bonds)

    # Benchmark original
    start = time.monotonic()
    for _ in range(3):
        _bond_features_original(token_array, atom_array)
    time_original = (time.monotonic() - start) / 3

    # Benchmark vectorized
    start = time.monotonic()
    for _ in range(3):
        _bond_features_vectorized(token_array, atom_array)
    time_vectorized = (time.monotonic() - start) / 3

    speedup = time_original / max(time_vectorized, 1e-9)
    print(f"\nBond features benchmark ({num_tokens} tokens, {num_tokens * atoms_per_token} atoms):")
    print(f"  original (O(N^2) loops): {time_original:.3f}s")
    print(f"  vectorized (NumPy):      {time_vectorized:.3f}s")
    print(f"  speedup:                 {speedup:.0f}x")

    assert time_vectorized < time_original, (
        f"Vectorized ({time_vectorized:.3f}s) should be faster than original ({time_original:.3f}s)"
    )


def test_design_tokens_excluded():
    """Verify design tokens are excluded from bond features."""
    DESIGN_RESIDUES = {"xpb": 32, "xpa": 33, "rbb": 34, "raa": 35}

    np.random.seed(42)
    token_array, atom_array = _make_mock_data(20, 3, 30)

    # Mark first token as a design residue
    design_res = list(DESIGN_RESIDUES)[0] if DESIGN_RESIDUES else "xpb"
    first_token_atoms = token_array[0].atom_indices
    for a in first_token_atoms:
        atom_array.res_name[a] = design_res

    result = _bond_features_vectorized(token_array, atom_array)
    # Design token should have no bonds
    assert result[0, :].sum() == 0, "Design token should have no outgoing bonds"
    assert result[:, 0].sum() == 0, "Design token should have no incoming bonds"
