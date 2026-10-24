from rdkit.Chem.rdchem import BondStereo, BondType


BOND_FEATURE_DIM = 14
BOND_TYPES = (
    BondType.SINGLE,
    BondType.DOUBLE,
    BondType.TRIPLE,
    BondType.AROMATIC,
    BondType.DATIVE,
)
BOND_STEREO_TYPES = (
    BondStereo.STEREONONE,
    BondStereo.STEREOANY,
    BondStereo.STEREOZ,
    BondStereo.STEREOE,
    BondStereo.STEREOCIS,
    BondStereo.STEREOTRANS,
)


def bond_features(bond, reverse=False):
    """Return bond chemistry plus donor-to-acceptor direction for one edge."""
    bond_type = bond.GetBondType()
    stereo_type = bond.GetStereo()
    if bond_type not in BOND_TYPES:
        raise ValueError(f"Unsupported RDKit bond type: {bond_type}")
    if stereo_type not in BOND_STEREO_TYPES:
        raise ValueError(f"Unsupported RDKit bond stereo: {stereo_type}")

    return (
        *(float(bond_type == known_type) for known_type in BOND_TYPES),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        *(float(stereo_type == known_type) for known_type in BOND_STEREO_TYPES),
        float(bond_type == BondType.DATIVE and not reverse),
    )


def molecule_bond_features(molecule):
    """Return directed edge indices and aligned features for an RDKit molecule."""
    edge_index = [[], []]
    edge_attr = []
    for bond in molecule.GetBonds():
        source = bond.GetBeginAtomIdx()
        target = bond.GetEndAtomIdx()
        forward_features = bond_features(bond)
        reverse_features = bond_features(bond, reverse=True)
        edge_index[0].extend((source, target))
        edge_index[1].extend((target, source))
        edge_attr.extend((forward_features, reverse_features))
    return edge_index, edge_attr
