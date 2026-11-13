"""Construct both controllers and inspect metadata with physics stepping blocked."""
import argparse
from contextlib import ExitStack
import json
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    args = parser.parse_args()
    from .numerics import ensure_numeric_environment
    numerics = ensure_numeric_environment()
    import mujoco
    from cold_start_take_v1.runtime import make_session, identity
    checks = {}
    with ExitStack() as stack:
        for name in ("mj_step", "mj_step1", "mj_step2"):
            stack.enter_context(patch.object(mujoco, name, side_effect=AssertionError("Physics stepping is forbidden in verify")))
        for arm in ("baseline", "macro"):
            session = make_session(arm, "nominal", 12345, runtime_config=args.config)
            meta = session.metadata()
            checks[arm] = dict(
                tick_zero=session.tick == 0,
                sensor_samples_zero=session.sensors.samples == 0,
                time_zero=session.time == 0.,
                historical_admission_not_claimed=session.admission["passed"] is False,
                physical_acceptance_not_claimed=meta["new_environment_physical_acceptance"] == "not_run",
                no_pregrasp=meta["common_preparation_used"] is False,
            )
    if not all(all(row.values()) for row in checks.values()):
        raise AssertionError(checks)
    print(json.dumps(dict(
        software_construction_checks="passed", checks=checks,
        runtime_input_files=len(identity()), physics_steps=0, renderer_created=False,
        experiment_results_created=False, independent_physical_acceptance="not_run",
        numerics=numerics,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
