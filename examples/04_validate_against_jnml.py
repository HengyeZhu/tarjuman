"""Check a converted model against the reference NeuroML simulator.

    pip install pyneuroml          # needs a Java runtime
    python examples/04_validate_against_jnml.py NeuroML2/LEMSexamples/LEMS_NML2_Ex5_DetCell.xml

Agreement with analytic references says the equations were implemented
correctly. Agreement with ``jnml`` says the *model* was translated correctly,
which is the thing a NeuroML user needs to know.

Two simulators that agree perfectly still drift apart in spike timing, because
they discretise differently, so the comparison separates spike count, spike
timing and the sub-threshold voltage rather than taking a single RMS over the
whole trace. Run with ``--converge`` to see where each simulator lands as the
time step falls: a difference that shrinks is numerical, one that persists is
a disagreement.
"""

import sys
from pathlib import Path

import tarjuman
from tarjuman.validate import convergence_study, reference_available, validate_lems

FIXTURE = Path(__file__).parent.parent / "tests" / "data" / "LEMS_hh_single_comp.xml"


def main() -> None:
    if not reference_available():
        print(
            "jnml was not found. Install the reference simulator with\n"
            "    pip install pyneuroml\n"
            "(it needs a Java runtime)."
        )
        raise SystemExit(2)

    arguments = [a for a in sys.argv[1:] if not a.startswith("--")]
    model = Path(arguments[0]) if arguments else FIXTURE
    if not arguments:
        print(f"No LEMS file given; using the bundled fixture {model.name}.\n")

    result = validate_lems(model)
    print(result.summary())

    if "--converge" in sys.argv:
        print("\nConvergence of the mean inter-spike interval:")
        study = convergence_study(model, steps=(0.01, 0.005, 0.001))
        print(f"{'dt (ms)':>9}  {'reference':>12}  {'tarjuman':>12}  {'diff':>10}")
        for step, reference, candidate in zip(
            study["steps"], study["reference_isi"], study["tarjuman_isi"]
        ):
            if reference is None or candidate is None:
                print(f"{step:>9}  {'-':>12}  {'-':>12}  {'-':>10}")
                continue
            print(
                f"{step:>9}  {reference:>12.5f}  {candidate:>12.5f}  "
                f"{candidate - reference:>+10.5f}"
            )
        print(
            "\nThe reference integrates explicitly and Jaxley implicitly, so the\n"
            "two approach the exact solution from opposite sides. A difference\n"
            "that shrinks as dt falls is discretisation, not disagreement."
        )

    raise SystemExit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
