"""Closed-form counterexamples for the actual V10 allocator; no physics."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import v10.directional_allocator as subject


def fixture():
    response = np.zeros((25, 8)); response[24, 5] = 1.; response[0, 6] = 1.; response[1, 7] = 1.
    row = np.zeros((1, 8)); row[0, 0] = 1.
    return dict(nominal_increment8=np.zeros(8), step_lower8=np.full(8, -.5), step_upper8=np.full(8, .5),
        robot_free25=np.zeros(25), robot_response25x8=response, output_axes25=subject.ROBOT_AXES25,
        user_target3=np.zeros(3), tracking_tolerances3=[.01, .015, .015], empirical_robot_error25=np.zeros(25),
        contact_rows_mx8=row, contact_free_delta_m=np.zeros(1), contact_error_m=np.zeros(1), contact_urgency_m=np.ones(1),
        model_scope_label="SYNTHETIC_ANALYTIC_FIXTURE_NOT_PHYSICAL")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=ROOT / "review/v10_directional_allocator_pure_v1.json"); args = parser.parse_args()
    checks = []; examples = []
    def check(name, passed, **details):checks.append(dict(name=name, passed=bool(passed), **details))
    def run(name, kw):
        before = copy.deepcopy(kw); result = subject.DirectionalAllocator().solve(**kw)
        check(name + "/actual_bounds", np.all(result.increment8 >= kw["step_lower8"]) and np.all(result.increment8 <= kw["step_upper8"]))
        check(name + "/no_force_or_global_certificate", result.debug["pressure_safety_certified"] is False and result.debug["global_model_admitted"] is False and result.debug["force_safety_slack_N"] is None)
        check(name + "/inputs_not_modified", all(np.array_equal(v, before[k]) if isinstance(v, np.ndarray) else v == before[k] for k, v in kw.items()))
        examples.append(dict(name=name, increment8=result.increment8.tolist(), debug=result.debug)); return result
    kw = fixture(); kw["contact_rows_mx8"][0] = 0.; kw["contact_rows_mx8"][0, 5] = 1.
    kw["nominal_increment8"][[5, 6]] = [-.1, .2]; kw["user_target3"] = [-.1, .2, 0.]
    r = run("motion_itself_retreats", kw)
    check("away_motion_bitexact", np.array_equal(r.increment8, kw["nominal_increment8"]))
    kw = fixture(); kw["nominal_increment8"][[0, 6]] = [.1, .2]; kw["user_target3"] = [0., .2, 0.]
    r = run("only_inward_modified", kw)
    check("orthogonal_progress_retained", abs(r.increment8[0]) < 1e-7 and abs(r.increment8[6] - .2) < 1e-7, actual=r.increment8.tolist())
    kw = fixture(); kw["contact_rows_mx8"][0] = 0.; kw["contact_rows_mx8"][0, 5] = 1.
    kw["nominal_increment8"][[5, 6]] = [.2, .15]; kw["user_target3"] = [.2, .15, 0.]; kw["tracking_tolerances3"] = [.1, .015, .015]
    r = run("retreat_only_with_tube_room", kw)
    check("tube_room_boundary_oracle", abs(r.increment8[5] - .1) < 2e-7 and abs(r.increment8[6] - .15) < 2e-7)
    kw = fixture(); kw["contact_rows_mx8"][0] = 0.; kw["contact_rows_mx8"][0, 5] = 1.
    kw["nominal_increment8"][5] = .2; kw["user_target3"] = [2., 0., 0.]; kw["tracking_tolerances3"] = [0., .015, .015]
    r = run("original_tube_unreachable", kw)
    check("unreachable_goal_not_rewritten", r.debug["user_target3"] == [2., 0., 0.] and abs(r.increment8[5] - .5) < 1e-7)
    check("tube_priority_over_inward_geometry", abs(r.debug["tube_point_violation3"][0] - 1.5) < 1e-7 and abs(r.debug["unresolved_geometry_m"][0] - .5) < 1e-7)
    kw = fixture(); kw["step_lower8"][0] = 0.; kw["nominal_increment8"][0] = .1; kw["contact_free_delta_m"][0] = .1
    r = run("one_sided_no_away_authority", kw)
    check("one_sided_is_not_physical_infeasibility", abs(r.increment8[0]) < 1e-7 and r.debug["unresolved_geometry_m"][0] >= .1 - 1e-7 and r.debug["physical_infeasibility_certified"] is False)
    kw = fixture(); kw["step_lower8"][:] = 0.; kw["step_upper8"][:] = 0.; kw["contact_free_delta_m"][0] = .1
    r = run("all_inputs_fixed", kw); check("all_fixed_finite", np.array_equal(r.increment8, np.zeros(8)) and np.isfinite(r.predicted_robot25).all())
    kw = fixture(); kw["step_lower8"][0] = kw["step_upper8"][0] = .1; kw["step_lower8"][1] = 0.; kw["step_upper8"][1] = .2
    kw["nominal_increment8"][0] = .1; kw["contact_rows_mx8"][0, 1] = -1.
    r = run("coupled_saturated_input_removed", kw)
    check("coupled_allocation_oracle", r.increment8[0] == .1 and abs(r.increment8[1] - .1) < 1e-7)
    kw = fixture(); kw["nominal_increment8"][0] = -.1; kw["proposal_lower8"] = np.zeros(8); kw["proposal_upper8"] = np.full(8, .001)
    kw["proposal_l1_weights8"] = np.ones(8); kw["proposal_l1_budget"] = 0.
    r = run("healthy_nominal_ignores_model_proposal_box", kw)
    check("healthy_bitpass_before_model_domain", np.array_equal(r.increment8, kw["nominal_increment8"]))
    kw = fixture(); kw["robot_free25"] = kw["robot_response25x8"] = None; kw["nominal_increment8"][5] = .1
    r = run("missing_model_keeps_motion", kw)
    check("missing_model_no_fake_zero_gain_or_freeze", np.array_equal(r.increment8, kw["nominal_increment8"]) and r.predicted_robot25 is None and r.debug["status"].startswith("UNAVAILABLE"))
    kw = fixture(); kw["robot_free25"] = kw["robot_response25x8"] = None; kw["nominal_increment8"][0] = .7
    r = run("missing_model_legal_nominal", kw); check("only_real_bounds_clip_missing_model", r.increment8[0] == .5 and r.debug["nominal_real_bound_adjustment"])
    kw = fixture(); kw["nominal_increment8"][5] = .1; kw["contact_rows_mx8"] = None; kw["contact_free_delta_m"] = kw["contact_error_m"] = kw["contact_urgency_m"] = None
    unknown = run("unknown_contact_rows", kw)
    kw["contact_rows_mx8"] = np.empty((0, 8)); empty = run("known_empty_contact_rows", kw)
    check("unknown_and_empty_differ", unknown.debug["geometry_status"] != empty.debug["geometry_status"] and np.array_equal(unknown.increment8, empty.increment8))
    kw = fixture(); kw["contact_rows_mx8"][:] = 0.; kw["contact_free_delta_m"][0] = .1
    r = run("known_true_zero_geometry_row", kw)
    check("zero_gradient_unresolved_not_dropped", r.debug["zero_geometry_rows"] == [0] and abs(r.debug["unresolved_geometry_m"][0] - .1) < 1e-12)
    kw = fixture(); kw["contact_error_m"] = None; kw["nominal_increment8"][0] = .1
    r = run("unknown_contact_error", kw)
    check("unknown_error_not_fabricated_zero", np.array_equal(r.increment8, kw["nominal_increment8"]) and r.debug["geometric_empirical_residual_m"] is None)
    kw = fixture(); kw["empirical_robot_error25"] = None; kw["nominal_increment8"][0] = -.1
    r = run("unknown_robot_error_point_only", kw); check("robot_error_unknown_label", r.debug["tube_empirical_error_known"] is False and r.debug["tube_empirical_violation3"] is None)
    kw = fixture(); kw["nominal_increment8"][0] = .1; kw["proposal_lower8"] = np.full(8, .6); kw["proposal_upper8"] = np.full(8, .7)
    r = run("model_proposal_box_empty", kw)
    check("empty_model_box_keeps_real_nominal", np.array_equal(r.increment8, kw["nominal_increment8"]) and "UNAVAILABLE_MODEL_PROPOSAL" in r.debug["status"])
    kw = fixture(); kw["contact_free_delta_m"][0] = .1; kw["nominal_increment8"][0] = .25; kw["proposal_lower8"] = np.zeros(8); kw["proposal_lower8"][0] = .2
    kw["proposal_upper8"] = np.full(8, .5); kw["proposal_l1_weights8"] = np.ones(8); kw["proposal_l1_budget"] = .1
    r = run("model_proposal_l1_empty", kw); check("empty_l1_is_not_mechanical_freeze", np.array_equal(r.increment8, kw["nominal_increment8"]) and "L1_DOMAIN" in r.debug["status"])
    kw = fixture(); kw["nominal_increment8"][[0, 6]] = [.1, .2]; kw["contact_free_delta_m"][0] = .1; kw["user_target3"] = [0., .2, 0.]
    kw["proposal_l1_weights8"] = np.ones(8); kw["proposal_l1_budget"] = .15
    r = run("empirical_l1_search_constraint", kw)
    check("l1_and_tube_priority", np.sum(abs(r.increment8)) <= .15 + 1e-8 and abs(r.increment8[6] - .15) < 2e-7)
    kw = fixture(); kw["step_lower8"][:] = kw["step_upper8"][:] = 0.; kw["step_lower8"][0] = -1e-300; kw["step_upper8"][0] = 1e-300
    kw["contact_rows_mx8"][0, 0] = 1e300; kw["contact_free_delta_m"][0] = .1
    r = run("finite_tiny_action_normalization", kw)
    check("tiny_scale_is_finite", np.isfinite(r.increment8).all() and abs(r.increment8[0] + 1e-301) < 1e-307)
    kw = fixture(); kw["contact_urgency_m"][0] = 1e300; kw["nominal_increment8"][0] = .1
    r = run("urgency_common_scaling_finite", kw); check("urgency_scale_no_changed_minimizer", abs(r.increment8[0]) < 1e-7)
    original_qp = subject._qp
    try:
        subject._qp = lambda Q, q, A, b, seed: (np.full_like(seed, 999.), dict(optimal=True, residual=0., status="lying_mock"))
        kw = fixture(); kw["nominal_increment8"][0] = .1
        r = run("false_solver_residual_rejected", kw)
        check("original_matrix_recheck_catches_false_optimal", r.debug["status"].startswith("NUMERICAL_STAGE_REJECTED") and r.debug["lexicographic_optimal"] is False)
        subject._qp = lambda Q, q, A, b, seed: (seed.copy(), dict(optimal=False, residual=0., status="iteration_limit_feasible"))
        r = run("nonoptimal_not_renamed", kw); check("nonoptimal_distinct", r.debug["status"] == "CHECKED_NONOPTIMAL_STAGE")
    finally:subject._qp = original_qp
    bad = []
    kw = fixture(); kw["nominal_increment8"] = ["0"] * 8; bad.append(("numeric_strings", kw))
    kw = fixture(); kw["nominal_increment8"] = [False] + [0.] * 7; bad.append(("python_bool", kw))
    kw = fixture(); kw["contact_urgency_m"] = [np.bool_(True)]; bad.append(("numpy_bool", kw))
    kw = fixture(); kw["robot_response25x8"][0, 0] = np.nan; bad.append(("unknown_not_zero_gain", kw))
    kw = fixture(); kw["contact_error_m"][0] = -.1; bad.append(("negative_error", kw))
    kw = fixture(); kw["step_lower8"][0] = .6; bad.append(("bad_actual_domain", kw))
    kw = fixture(); kw["output_axes25"] = tuple(["object_x"] + list(subject.ROBOT_AXES25[1:])); bad.append(("object_axis", kw))
    for name, kw in bad:
        try:subject.DirectionalAllocator().solve(**kw)
        except ValueError:check("strict_reject_" + name, True)
        else:check("strict_reject_" + name, False)
    path = Path(subject.__file__)
    result = dict(schema="v10_directional_allocator_analytic_qa_v1", passed=all(c["passed"] for c in checks), checks=checks,
        examples=examples, allocator_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        qa_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), physics_steps=0,
        limitations="analytic synthetic allocator tests only, not plant effectiveness, pressure prediction or model admission")
    with args.output.open("x", encoding="utf-8") as stream:json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(passed=result["passed"], checks=len(checks), failed=[c["name"] for c in checks if not c["passed"]], source=result["allocator_source_sha256"], physics_steps=0), ensure_ascii=False))
    if not result["passed"]:raise SystemExit(1)


if __name__ == "__main__":main()
