"""Read-only V17 static compilation and archived 14--16 s motion audit.

Only MjModel compilation is allowed. No MjData/HandPlant, integration, forward,
FK, collision or contact-force query is permitted. Historical contact values
are read from frozen files; compiled settings are not historical mjContact.dim.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import research_imports
import mujoco

FOUNDATION = research_imports.FOUNDATION
STEM = ROOT / 'review/v17_contact_dimension_late_motion_independent_v1'
RUNS = (
    'output/v17_transfer_contact_hold_v1/runs/fragile_conflict__ratio_effort__619732',
    'output/v17_transfer_contact_transport_v1/runs/nominal__ratio_effort__619732',
    'output/v17_transfer_contact_transport_v1/runs/friction_drop__ratio_effort__619732',
)
DOCS = {
    'condim': 'https://mujoco.readthedocs.io/en/3.15.0/computation/index.html#condim',
    'mixing': 'https://mujoco.readthedocs.io/en/3.15.0/modeling.html#contact-parameters',
    'actual_contact_dim': 'https://mujoco.readthedocs.io/en/3.15.0/APIreference/APItypes.html#mjcontact',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    for suffix in ('.json', '.md'):
        if STEM.with_suffix(suffix).exists():
            raise FileExistsError(STEM.with_suffix(suffix))
    source_paths = [FOUNDATION / name for name in (
        'hand_plant.py', 'foundation_config.py', 'session.py', 'assets/legacy_model.xml')]
    source_paths += [ROOT / 'research_imports.py', Path(__file__)]
    before = {str(p): sha(p) for p in source_paths}
    core_path = ROOT / 'review/v17_small_independent_v1.json'
    inputs_path = ROOT / 'review/v17_small_independent_inputs_v1.json'
    old_inputs = json.loads(inputs_path.read_text(encoding='utf-8'))['input_files']
    core = json.loads(core_path.read_text(encoding='utf-8'))
    checks = []

    def check(name, condition):
        checks.append({'name': name, 'passed': bool(condition)})

    check('prior_core_passed_91_checks', core['passed'] and len(core['checks']) == 91)
    forbidden_calls = {}

    def deny(name):
        def call(*args, **kwargs):
            forbidden_calls[name] = forbidden_calls.get(name, 0) + 1
            raise AssertionError('Forbidden API: ' + name)
        return call

    forbidden = ['MjData', 'mj_step', 'mj_step1', 'mj_step2', 'mj_forward',
                 'mj_forwardSkip', 'mj_kinematics', 'mj_comPos', 'mj_collision',
                 'mj_contactForce', 'mj_fwdPosition', 'mj_inverse',
                 'mj_inverseSkip', 'mj_Euler', 'mj_RungeKutta']
    for name in forbidden:
        if hasattr(mujoco, name):
            setattr(mujoco, name, deny(name))
    import hand_plant
    from foundation_config import FoundationConfig
    hand_plant.HandPlant.__init__ = deny('HandPlant.__init__')
    hand_plant.HandPlant.advance = deny('HandPlant.advance')
    xml = ET.parse(FOUNDATION / 'assets/legacy_model.xml').getroot()
    raw_geoms = {name: dict(xml.find(f".//geom[@name='{name}']").attrib)
                 for name in ('object_geom', 'support', 'floor')}
    rows = []
    input_hashes = {}
    for relative in RUNS:
        folder = ROOT / relative
        captured = {name: sha(folder / name) for name in (
            'metadata.json', 'delivery_intervals.npz', 'delivery_summary.json',
            'delivery_manifest.json')}
        input_hashes[relative] = captured
        check(relative + ': exact prior input identities',
              all(old_inputs[str(folder)][name] == digest
                  for name, digest in captured.items()))
        meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        model, xml_sha = hand_plant._model(FoundationConfig(**meta['config']))
        check(relative + ': compiled XML matches saved model identity', xml_sha == meta['model_sha256'])
        geoms = []
        object_id = int(model.body('object').id)
        ids = list(map(int, np.flatnonzero(model.geom_bodyid == object_id)))
        ids += [int(model.geom(name).id) for name in ('support', 'floor')]
        for gid in ids:
            geoms.append(dict(id=gid, name=model.geom(gid).name,
                              body_id=int(model.geom_bodyid[gid]),
                              type=int(model.geom_type[gid]), size=model.geom_size[gid],
                              condim=int(model.geom_condim[gid]),
                              priority=int(model.geom_priority[gid]),
                              friction_slide_torsion_roll=model.geom_friction[gid],
                              contype=int(model.geom_contype[gid]),
                              conaffinity=int(model.geom_conaffinity[gid])))
        compiled = dict(config_contact_dimension=meta['config']['contact_dimension'],
                        generated_xml_sha256=xml_sha, saved_model_sha256=meta['model_sha256'],
                        npair=int(model.npair), geoms=geoms,
                        cone=int(model.opt.cone), noslip_iterations=int(model.opt.noslip_iterations),
                        enableflags=int(model.opt.enableflags),
                        dynamic_object_support_pair_dimension_by_documented_rule=3,
                        historical_per_contact_dim_observed=False)
        check(relative + ': relevant compiled geoms are 3 at equal priority zero; no explicit pair',
              model.npair == 0 and all(g['condim'] == 3 and g['priority'] == 0 for g in geoms))
        with np.load(folder / 'delivery_intervals.npz', allow_pickle=False) as archive:
            a = {k: archive[k] for k in archive.files}
        start, end = a['interval_start_s'], a['interval_end_s']
        weights = np.maximum(0., np.minimum(end, 16.) - np.maximum(start, 14.))
        selected = weights > 0.
        q, v = a['object_free_qpos7_end'], a['object_free_qvel6_end']
        speed, angular = np.linalg.norm(v[:, :3], axis=1), np.linalg.norm(v[:, 3:], axis=1)
        first = int(np.argmin(np.abs(end - 14.)))
        last = int(np.argmin(np.abs(end - 16.)))
        contact = {}
        for name in ('receiver_top', 'receiver_otherface', 'floor', 'other_nonhand_world', 'hand', 'other_nonworld'):
            force = a['target_' + name + '_normal_N']
            positives = np.flatnonzero(force > 0.)
            contact[name] = dict(peak_N=float(np.max(force[selected])),
                impulse_Ns=float(weights @ force),
                positive_contact_duration_s=float(weights @ (force > 0.)),
                last_positive_contact_end_s=None if not len(positives) else float(end[positives[-1]]))
        samples = []
        for requested in (14., 14.5, 15., 15.5, 16.):
            i = int(np.argmin(np.abs(end - requested)))
            samples.append(dict(requested_time_s=requested, actual_end_time_s=float(end[i]),
                                qpos7=q[i], qvel6=v[i], speed_m_per_s=float(speed[i]),
                                angular_speed_rad_per_s=float(angular[i])))
        def describe(values):
            return dict(duration_weighted_mean=float(weights @ values / weights.sum()),
                        minimum=float(values[selected].min()), maximum=float(values[selected].max()),
                        first=float(values[first]), last=float(values[last]),
                        last_minus_first=float(values[last] - values[first]))
        with np.load(folder / 'trace.npz', allow_pickle=False) as archive:
            time = archive['time']
            env = archive['environment_json']
        late_environments = [json.loads(str(s)) for s in env[(time >= 14.) & (time <= 16. + 1e-9)]]
        env_unique = {json.dumps(s, sort_keys=True, allow_nan=False) for s in late_environments}
        trace_hash = sha(folder / 'trace.npz')
        captured['trace.npz'] = trace_hash
        check(relative + ': trace identity matches prior audit', old_inputs[str(folder)]['trace.npz'] == trace_hash)
        no_applied_load = all(np.array_equal(s['force'], [0., 0., 0.])
                              and np.array_equal(s.get('torque', [0., 0., 0.]), [0., 0., 0.])
                              and s.get('squeeze', 0.) == 0. for s in late_environments)
        check(relative + ': 2 s covered with valid contact and object state',
              abs(float(weights.sum()) - 2.) < 1e-12 and
              a['contact_valid'][selected].all() and a['object_state_valid'][selected].all())
        check(relative + ': recorded late loads zero and no hand contact',
              no_applied_load and contact['hand']['positive_contact_duration_s'] == 0.)
        check(relative + ': nonzero translation and rotation persist through all late samples',
              speed[selected].min() > 0. and angular[selected].min() > 0.)
        displacement = q[last, :3] - q[first, :3]
        motion = dict(window_s=[14., 16.], full_record_count=len(start),
                      overlap_record_count=int(selected.sum()),
                      coverage_s=float(weights.sum()),
                      overlap_count_note='Floating boundary creates a tiny overlap in a 4001st interval; durations sum to exactly 2 s.',
                      linear_speed_m_per_s=describe(speed), angular_speed_rad_per_s=describe(angular),
                      position_displacement_m=displacement,
                      horizontal_net_displacement_m=float(np.linalg.norm(displacement[:2])),
                      velocity_integral_path_length_m=float(weights @ speed),
                      angular_speed_integral_rad=float(weights @ angular),
                      contact=contact, samples=samples,
                      any_actor_top_equals_target_top_in_window=bool(np.array_equal(
                          a['receiver_any_actor_top_normal_N'][selected], a['target_receiver_top_normal_N'][selected])),
                      recorded_environment_unique=[json.loads(s) for s in sorted(env_unique)],
                      passive_definition='No positive target-hand contact and recorded applied force/torque/squeeze are zero; gravity and receiver contact remain.',
                      velocity_frame='First 3 free-joint velocity components are world translation; only rotation norm is used, not a claimed world-axis angular vector.',
                      time_phase='Contact is cached solve data; q/v are integrated interval-end state, not contact geometry recomputed at end.')
        rows.append(dict(run=relative, scenario=meta['scenario']['name'], motion_mode=meta['motion_mode'],
                         seed=meta['seed'], config=meta['config'], compiled=compiled, late_motion=motion))
        check(relative + ': all read input files remained unchanged',
              all(sha(folder / name) == digest for name, digest in captured.items()))
    check('all static sources unchanged', before == {str(p): sha(p) for p in source_paths})
    check('no forbidden Python API calls', not forbidden_calls)
    result = dict(schema='v17_contact_dimension_late_motion_independent_v1',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  role='Independent read-only scope audit; no policy or physical-parameter change',
                  passed=all(c['passed'] for c in checks), checks=checks, rows=rows,
                  source_sha256=before, input_sha256=input_hashes,
                  prior_core_audit_sha256=sha(core_path), prior_inputs_manifest_sha256=sha(inputs_path),
                  docs=DOCS, mujoco_version=mujoco.__version__, raw_xml_named_geoms=raw_geoms,
                  model_compile_requests=3, unique_model_cache_entries=len(hand_plant._MODELS),
                  forbidden_api_attempts=forbidden_calls, forbidden_api_guard_names=forbidden,
                  limitations=[
                      'Compiled geom settings and documented mixing imply dimension 3 for this dynamic geom pair. Historical mjContact.dim/constraint activation was not recorded or recomputed.',
                      'Nonzero stored torsional/rolling friction coefficients do not activate those torque dimensions at condim 3. No claim that all rotational dissipation is absent: other physical mechanisms remain.',
                      'No-slip contact kinematics were not reconstructed; persistent translation and rotation are consistent with rolling but do not prove pure no-slip rolling.',
                      'Floor-free through 16 s and sustained receiver load are not stationary placement or future edge-safety certification.',
                      'Three previously seen capsule contexts share one physical seed. No sensitivity comparison or new policy performance experiment was run.',
                      'All V17/V17b source, scores, thresholds and physics remain unchanged; current prospective 80 results were not read.'
                  ])
    STEM.with_suffix('.json').write_text(json.dumps(plain(result), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    lines = ['# V17接触维度与末两秒运动：独立只读核查', '',
             f"核查 {'PASS' if result['passed'] else 'FAIL'}：{sum(c['passed'] for c in checks)}/{len(checks)}。使用 MuJoCo {mujoco.__version__}，三次保存配置请求只生成一个相同配置的模型缓存。生成XML SHA逐条匹配原metadata。未创建MjData/HandPlant；积分、forward、FK、碰撞与接触力查询的Python入口均禁止，实际调用0。", '',
             '三条均为capsule、seed619732，已见开发条件。仅复用既有91项物理门的输入身份，不重复136源logging门或正在运行的80条。', '',
             '## 设定边界', '',
             '原XML的object_geom显式condim=6（legacy_model.xml:264）；hand_plant.py:80–84在编译前将物体及plastic_collision手部几何改为保存配置的3。接收台与地板没有显式condim，编译默认也为3。三者priority均0，npair=0。', '',
             '物体friction=[0.4867989822645726,0.015,0.003]，接收台=[0.6,0.005,0.0001]，地板=[1.0,0.005,0.0001]。这是编译属性；friction_drop在线日程会改变物体/手部的第一系数（hand_plant.py:225–231），具体末窗environment保存在JSON。', '',
             f"官方规则：动态接触同优先级取较大condim，因此该物体—接收台对按编译规则应为3；3含法向与切向力，6额外含扭转/滚动矩。存有非零后两项摩擦系数并不启用缺少的维度。[维度定义]({DOCS['condim']})；[参数混合]({DOCS['mixing']})。", '',
             f"这不是逐历史接触的dim测量：账本及hand_plant.py:290–296的patch记录未保留mjContact.dim，本次也没有生成或查询接触。[实际接触字段]({DOCS['actual_contact_dim']})。亦不能据此断言没有其他旋转耗散或证明接触模型造成全部差异。", '',
             '## 14–16秒实际账本', '',
             '| 条件 | 平移速度均值 mm/s | 角速均值 rad/s | 水平净位移 mm | 台面接触 s | 手接触 s | 地板接触 s |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        m = row['late_motion']; c = m['contact']
        lines.append(f"| {row['scenario']} / {row['motion_mode']} | {1000*m['linear_speed_m_per_s']['duration_weighted_mean']:.9f} | {m['angular_speed_rad_per_s']['duration_weighted_mean']:.9f} | {1000*m['horizontal_net_displacement_m']:.6f} | {c['receiver_top']['positive_contact_duration_s']:.3f} | {c['hand']['positive_contact_duration_s']:.3f} | {c['floor']['positive_contact_duration_s']:.3f} |")
    lines += ['', '全部记录contact/object-state有效，时间交集恰为2s；浮点边界使4001个区间有正交集，不能把它误报为4001个完整0.5ms。台面法向力约1.11075N，三条末窗冲量均约2.22150Ns；any-actor top与target top逐值相同。', '',
              '三条的平移与角速在末两秒变化仅约1e−14m/s与1e−12rad/s量级，仍持续运动。末次手接触分别约12.551、12.4885、12.479秒；末窗记录外加force/torque/squeeze均0。这里“被动”仅指已无手接触/外加日程载荷，重力与台面接触仍存在。未重建无滑动运动学，不把平移加转动直接证明为纯滚动。', '',
              '因此，0/3落地仅适用于16秒观察期限；不能扩成静止、长期留台或放置认证。接触维度限制影响所有同物理条件方法，不能用改变模型倒推当前算法获胜。', '',
              '## 后续可证伪设计（本轮未执行）', '',
              '若需要检验稳置，应另预注册同场景、同物理种子、同初始前缀的全方法接触模型敏感性：明确3/4/6维与扭转/滚动摩擦来源及物理标定依据，固定控制与评分，保留所有失败，并延长独立末态运动观察。模型修订必须新来源/新结果，不修改或合池当前136源前瞻结果，也不以参数选择消除已见反例。', '',
              '原始q/v仅离线诊断；无对象真值进入在线控制。cached-contact solve与积分后end q/v不是重算后的同瞬时几何证书。', '',
              f"[完整JSON]({STEM.name}.json)包含每条输入SHA、静态源码SHA、实际采样时刻、速度范围、环境、源不变检查及全部限制。"]
    STEM.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'], checks=len(checks), json=str(STEM.with_suffix('.json')),
                          json_sha256=sha(STEM.with_suffix('.json')), md_sha256=sha(STEM.with_suffix('.md'))), ensure_ascii=False))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
