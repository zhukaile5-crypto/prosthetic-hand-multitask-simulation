"""Finite algebra + archived public-input author checks. No new physics."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy, copy
from dataclasses import asdict
import argparse, hashlib, json, sys
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import research_imports
import mujoco
from v20_contact_exit.geometry import position_reference_map, contact_geometry_rows
from v20_contact_exit.adapter import ContactExitAdapter
from v13_a1.codec import Reference21
from v16_release_geometry.public_adapter import PublicReleaseAdapter


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def rejects(fn):
    try: fn()
    except (ValueError, TypeError): return True
    return False


def main(out):
    if out.exists(): raise FileExistsError('Fresh author evidence required')
    checks = []
    def ck(name, condition, details=None):
        checks.append(dict(name=name, passed=bool(condition), details=details))
    files = ['v20_contact_exit/geometry.py', 'v20_contact_exit/adapter.py']
    initial = {f: sha(ROOT / f) for f in files}
    forbidden = []
    def stop(*a, **kw):
        forbidden.append('physical_pipeline'); raise RuntimeError('No integration/forward/contact evaluation allowed')
    for name in ('mj_step', 'mj_step1', 'mj_step2', 'mj_forward', 'mj_collision', 'mj_contactForce'):
        if hasattr(mujoco, name): setattr(mujoco, name, stop)
    A = np.zeros((21, 25)); A[:, :21] = np.eye(21)
    G = np.zeros((21, 3)); G[:, 0] = np.arange(1., 22.)
    bias = np.zeros((21, 3)); bias[:, 1] = -G[:, 0]
    kw = dict(transmission21x25=A, joint_scale25=np.ones(25), gain21x3=G,
              bias21x3=bias, fixed_gain21=np.ones(21, bool), affine_bias21=np.ones(21, bool))
    B, diag = position_reference_map(**kw)
    ck('D1 right inverse has exact simple oracle and no response certificate',
       np.array_equal(B[:21], np.eye(21)) and np.array_equal(A @ B, np.eye(21)) and not diag['finite_servo_response_prediction'])
    bad = dict(kw); bad['transmission21x25'] = A.copy(); bad['transmission21x25'][1] = A[0]
    ck('rank deficient geometry rejected rather than zero filled', rejects(lambda: position_reference_map(**bad)))
    bad = dict(kw); bad['bias21x3'] = bias.copy(); bad['bias21x3'][0, 1] *= .5
    ck('non D1 affine motor law rejected', rejects(lambda: position_reference_map(**bad)))
    bad = dict(kw); bad['joint_scale25'] = np.ones(25); bad['joint_scale25'][0] = 0
    ck('zero metric and numeric strings rejected', rejects(lambda: position_reference_map(**bad)) and
       rejects(lambda: position_reference_map(**dict(kw, gain21x3=G.astype(str)))))
    J = np.zeros((3, 25)); J[0, 0] = 2.; J[1, 1] = 3.
    patch = dict(old_normal_world3=np.array([1., 0, 0]), jacobian3x25=J, normal_N=2., digit=0,
                 patch_index=0, acquisition_time_s=.01, robot_time_s=.02)
    debt = np.zeros(21); debt[0] = .003
    vel = np.zeros(25); vel[0] = .02
    rows = contact_geometry_rows(proposal_map25x21=B, current_unbiased_debt21=debt, robot_velocity25=vel, patches=[patch])
    ck('normal sign, current velocity and debt independent closed oracle',
       np.array_equal(rows['Npx21'][0], 2 * np.eye(21)[0]) and rows['vnormal_p'][0] == .04 and rows['a_normal_p'][0] == .006)
    split = [dict(patch, normal_N=1.), dict(patch, normal_N=1., patch_index=1)]
    rs = contact_geometry_rows(proposal_map25x21=B, current_unbiased_debt21=debt, robot_velocity25=vel, patches=split)
    u = np.arange(21) * .001; target = -.004
    def loss(r): return float(np.sum(r['y_p'] / r['y_p'].sum() * (r['Npx21'] @ u - target) ** 2))
    ck('pressure-share normal task is patch-splitting invariant', loss(rows) == loss(rs))
    opposite = contact_geometry_rows(proposal_map25x21=B, current_unbiased_debt21=debt, robot_velocity25=vel,
        patches=[patch, dict(patch, old_normal_world3=np.array([-1., 0, 0]), patch_index=1)])
    ck('opposing normals retain two rows instead of zero average', len(opposite['Npx21']) == 2 and np.linalg.norm(opposite['Npx21'][0]) > 0 and np.array_equal(opposite['Npx21'][1], -opposite['Npx21'][0]))
    from hand_plant import _model
    from foundation_config import FoundationConfig, OPEN
    model, _ = _model(FoundationConfig())
    trace = ROOT / 'output/v19_small_nominal_transport_v1/runs/nominal__ratio_effort__619741/trace.npz'
    with np.load(trace, allow_pickle=False) as z:
        time = z['time']; observation = z['observation_json']
    indices = np.flatnonzero((time >= 11.75 - 1e-9) & (time <= 12.05 + 1e-9))
    frames = [SimpleNamespace(**json.loads(str(observation[i]))) for i in indices]
    adapter = ContactExitAdapter(model, open18=np.concatenate(OPEN))
    parent = PublicReleaseAdapter(model, open18=np.concatenate(OPEN))
    support_same = True
    for frame in frames:
        ah = adapter.observe(frame); ph = parent.observe(frame)
        support_same &= asdict(ah) == asdict(ph)
    ck('archived public stream keeps parent signed hand evidence unchanged', support_same and len(frames) == 61)
    f = frames[-1]; current = np.asarray(f.robot_reference21)
    # Use actual legal software bounds intersected with native control range.
    lower = np.asarray(f.reference_lower21).copy(); upper = np.asarray(f.reference_upper21).copy()
    native_lo = adapter.actor.ctrl_range21[:, 0].copy(); native_hi = adapter.actor.ctrl_range21[:, 1].copy()
    native_lo[18] -= adapter.actor.lift_bias_m; native_hi[18] -= adapter.actor.lift_bias_m
    lower = np.maximum(lower, native_lo); upper = np.minimum(upper, native_hi)
    params = dict(current21=current, reference_lower21=lower, reference_upper21=upper)
    got = adapter.current_contact_tasks(f, **params)
    ck('real public packet admits explicit current reference geometry', got['available'], got['diagnostics'])
    if got['available']:
        ck('real source aliases and array shapes agree', got['Npx21'].shape[1] == 21 and got['normal_jacobian_Px25'].shape[1] == 25 and
           got['B25x21'] is got['proposal_map25x21'] and got['pressure_P_N'] is got['y_p'])
        ck('actor lift gravity bias is absent from geometric debt', got['current_unbiased_debt21'][18] == current[18] - got['encoder_reference21'][18] and adapter.actor.lift_bias_m > 0)
        ck('normal contributions use full current robot velocity/debt once',
           np.allclose(got['vnormal_p'], got['normal_jacobian_Px25'] @ np.asarray(f.robot_velocity25), atol=1e-15, rtol=0) and
           np.allclose(got['a_normal_p'], got['Npx21'] @ got['current_unbiased_debt21'], atol=1e-15, rtol=0))
        ck('real motion is at most one pair per digit, in m/s before dt',
           got['diagnostics']['motion_available'] and got['motion_Mx21'].shape[0] <= 10 and
           got['motion_Mx21'].shape[0] == len(got['motion_e_M']) and got['diagnostics']['motion']['error_units'] == 'metres per second')
        first = got['diagnostics']['motion']['digit_details'][0]
        T = np.asarray(f.tangent_basis)[first['digit']]
        old = adapter._motion_samples[int(round(f.sample_time / .005))]
        # Independent e sign/unit oracle uses already-recorded acquired robot projection + original device slip.
        oracle = np.asarray(first['acquired_robot_projection_m_s']) + np.asarray(f.slip_local)[first['digit']]
        ck('measured external absolute projection equals robot plus slip without dt', np.allclose(got['motion_e_M'][:2], oracle, atol=1e-15, rtol=0))
    broken = deepcopy(f); broken.slip_domain_acquisition_time_s -= .005
    missing = adapter.current_contact_tasks(broken, **params)
    ck('misaligned motion is unknown empty task, not zero measured motion', missing['available'] and not missing['diagnostics']['motion_available'] and missing['motion_Mx21'].shape == (0, 21))
    ck('unknown solver motion fields omitted together', missing['motion_matrix_Mx21'] is None and missing['motion_error_M_m_s'] is None and missing['motion_time_s'] is None)
    partial = deepcopy(f); partial.valid = np.asarray(f.valid).copy(); partial.valid[0] = False
    partial_result = adapter.current_contact_tasks(partial, **params)
    ck('partial hand validity retains other measured rows without all-hand zero fiction', partial_result['available'] and 0 not in partial_result['digits_p'] and partial_result['diagnostics']['partial_geometry'])
    unknown = deepcopy(f); unknown.valid = np.zeros(5, bool)
    ck('all missing geometry cannot masquerade as known empty contact', not adapter.current_contact_tasks(unknown, **params)['available'])
    empty = deepcopy(f); empty.skin_patches = []; empty.valid = np.ones(5, bool); empty.distributed_valid = np.ones(5, bool)
    empty_result = adapter.current_contact_tasks(empty, **params)
    ck('explicit valid zero-patch instrument differs from missing geometry', empty_result['available'] and empty_result['Npx21'].shape == (0, 21) and not empty_result['diagnostics']['normal_task_available'])
    changed = deepcopy(f); changed.robot_actuator_target21 = np.asarray(f.robot_actuator_target21).copy(); changed.robot_actuator_target21[0] += .01
    ck('engine/reference mismatch rejected', not adapter.current_contact_tasks(changed, **params)['available'])
    clipped = deepcopy(f); clipped.robot_reference21 = current.copy(); clipped.robot_reference21[0] = native_hi[0] + .01
    clipped.robot_actuator_target21 = adapter.actor.encode(Reference21(clipped.robot_reference21, f.time)).values
    ck('current native control clipping is unknown geometry', not adapter.current_contact_tasks(clipped, **dict(params, current21=clipped.robot_reference21))['available'])
    force_model = copy(model)
    ids = adapter.servo.aids
    force_model.actuator_forcelimited[ids] = True
    force_model.actuator_forcerange[ids, 0] = -1e-12; force_model.actuator_forcerange[ids, 1] = 1e-12
    force_adapter = ContactExitAdapter(force_model, open18=np.concatenate(OPEN))
    for frame in frames: force_adapter.observe(frame)
    force_result = force_adapter.current_contact_tasks(f, **params)
    ck('native force clipping retains target-length geometry and exact native diagnostic', force_result['available'] and any(force_result['native_state']['diagnostics']['force_clipped21']))
    bad_model = copy(model); bad_model.actuator_biasprm[ids[0], 1] *= .5
    bad_adapter = ContactExitAdapter(bad_model, open18=np.concatenate(OPEN))
    for frame in frames: bad_adapter.observe(frame)
    ck('actual-model unsupported target-length law unavailable', not bad_adapter.current_contact_tasks(f, **params)['available'])
    split_adapter = ContactExitAdapter(model, open18=np.concatenate(OPEN))
    split_frames = deepcopy(frames)
    for frame in split_frames:
        if frame.skin_patches:
            p = deepcopy(frame.skin_patches[0]); p['normal'] *= .5
            frame.skin_patches = [p, deepcopy(p)] + frame.skin_patches[1:]
        split_adapter.observe(frame)
    sg = split_adapter.current_contact_tasks(split_frames[-1], **params)
    ck('actual public split preserves coacquired digit motion task', got['available'] and sg['available'] and
       np.allclose(got['motion_Mx21'], sg['motion_Mx21'], atol=2e-14, rtol=0) and np.allclose(got['motion_e_M'], sg['motion_e_M'], atol=2e-14, rtol=0))
    ck('no actor writer, integration, forward or contact queries', not forbidden and adapter.actor.write_count == 0 and force_adapter.actor.write_count == 0)
    final = {f: sha(ROOT / f) for f in files}
    ck('owned source unchanged during author gate', initial == final)
    report = dict(schema='v20_geometry_adapter_author_v1', passed=all(c['passed'] for c in checks),
        created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=final,
        archived_input=dict(path=str(trace.relative_to(ROOT)), sha256=sha(trace), public_frames=len(frames)),
        checks=checks, checks_count=len(checks), issues=[c['name'] for c in checks if not c['passed']],
        scope='Finite algebra and archived public observations; known model compile/private robot FK/transmission only; zero integration/forward/contact-force/sensor calls; no performance evidence',
        physical_steps=0, forbidden_calls=forbidden)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'], checks=len(checks), issues=report['issues'], path=str(out), sha256=sha(out)), ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); raise SystemExit(main(args.out))
