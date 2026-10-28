from __future__ import annotations

import re

ATOMIC_NUMBER = {
    "H":1,"Li":3,"B":5,"C":6,"N":7,"O":8,"F":9,"Na":11,"Mg":12,"Al":13,
    "Si":14,"P":15,"S":16,"Cl":17,"K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,
    "Cr":24,"Mn":25,"Fe":26,"Co":27,"Ni":28,"Cu":29,"Zn":30,"Ga":31,"Ge":32,
    "As":33,"Se":34,"Br":35,"Rb":37,"Sr":38,"Y":39,"Zr":40,"Nb":41,"Mo":42,
    "Ru":44,"Rh":45,"Pd":46,"Ag":47,"Cd":48,"In":49,"Sn":50,"Sb":51,"Te":52,
    "I":53,"Cs":55,"Ba":56,"La":57,"Ce":58,"Pr":59,"Nd":60,"Sm":62,"Eu":63,
    "Gd":64,"Tb":65,"Dy":66,"Ho":67,"Er":68,"Tm":69,"Yb":70,"Lu":71,"Hf":72,
    "Ta":73,"W":74,"Re":75,"Os":76,"Ir":77,"Pt":78,"Au":79,"Hg":80,"Tl":81,
    "Pb":82,"Bi":83,"Th":90,"U":92
}


def parse_composition(formula):
    if not isinstance(formula, str) or not formula:
        return {}
    parts = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", formula)
    if not parts or "".join(e+n for e,n in parts) != formula or any(e not in ATOMIC_NUMBER for e, _ in parts):
        return {}
    out = {}
    for element, amount in parts:
        out[element] = out.get(element, 0.0) + float(amount or 1)
    return out


def composition_features(formula):
    amounts = parse_composition(formula)
    total = sum(amounts.values())
    known = [ATOMIC_NUMBER[e] for e in amounts if e in ATOMIC_NUMBER]
    return {
        "n_elements": len(amounts) or None,
        "total_atoms": total or None,
        "mean_atomic_number": (sum(ATOMIC_NUMBER[e] * n for e,n in amounts.items() if e in ATOMIC_NUMBER) / total) if total and len(known) == len(amounts) else None,
        "unknown_element": bool(amounts) and any(e not in ATOMIC_NUMBER for e in amounts),
        "element_fractions": {e: n / total for e,n in amounts.items()} if total else {},
    }
