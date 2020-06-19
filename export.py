import os

def serialize_qs(mp):
    structure = []
    for i, (ctl, pauli, trg) in enumerate(mp.model.structure):
        if len(ctl) == 0:
            ctl = "new Int[0]"
        structure.append(f"ControlledRotation(({trg}, {ctl}), Pauli{pauli}, {i}),")
    structure = ("\n" + " "*16).join(structure)
    return f"""
namespace Solution {{
    open Microsoft.Quantum.MachineLearning;
    
    // Cost {mp.cost}
    operation Solve () : (ControlledRotation[], (Double[], Double)) {{
        return (
            [
                {structure}
            ],
            ({mp.params[:-1].tolist()}, {mp.params[-1]})
        );
    }}
}}
"""

def write_qs(mp, fname="solution.qs", overwrite=False):
    if not overwrite:
        original, ext = os.path.splitext(fname)
        i = 0
        while os.path.exists(os.path.abspath(fname)):
            i += 1
            fname = f"{original}-{i}{ext}"
    with open(fname, "w") as f:
        f.write(serialize_qs(mp))
    print(f"Wrote model Q# to {fname}")
