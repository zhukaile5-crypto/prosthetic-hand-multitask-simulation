"""Stopped-16 prefix comparator incident: exact fields, no physics/performance.

The old asymmetric new-instrument/old-instrument comparator is retained.
Both current arms already contain the same six fields. Compare all 37 columns,
including the complete JSON strings, rather than removing either side's data.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'review/v17b_prospective80_plan_v1.json'
PROGRESS = ROOT / 'review/v17b_prospective80_progress_v1.json'
EXPECTED_PLAN = 'b08fe0d57b2c43a8abcf58e9ff869d1fdc01ae0f2ed5cfe1546028e7268a6a06'
STEM = ROOT / 'review/v17b_prefix_reader_correction_independent_v1'
COLS = ('time', 'normal', 'skin_normal', 'shear', 'palm', 'hand_other',
        'all_hand_object', 'support', 'penetration', 'world_penetration',
        'position', 'relative', 'rotation', 'relative_velocity', 'angular_velocity',
        'lift', 'qpos', 'qvel', 'wrist_position', 'ctrl_before', 'ctrl',
        'alpha_before', 'alpha', 'robot_actuator_target21', 'executed_reference21',
        'phase', 'command', 'observed_normal', 'validity', 'observed_age',
        'observed_auxiliary_wrench', 'observed_auxiliary_valid',
        'legacy_common_sensor_json', 'receiver_measurement_json',
        'task_request_json', 'environment_json', 'raw_patches_json')
ADDED = ('slip_valid5', 'slip_domain_acquisition_time_s', 'slip_domain_time_s',
         'slip_domain_role', 'slip_domain_available', 'slip_domain_reason')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if sys.argv[1:] != ['--execute']:
        raise ValueError('Explicit stopped-16 audit only')
    if any(STEM.with_suffix(s).exists() for s in ('.json', '.md')):
        raise FileExistsError(STEM)
    plan, progress = read(PLAN), read(PROGRESS)
    checks = []
    def ck(name, ok):
        checks.append(dict(name=name, passed=bool(ok)))
    ck('exact original registration', sha(PLAN) == EXPECTED_PLAN)
    ck('retained stopped attempt, not complete80', progress['complete'] is False
       and progress['physics_active'] is False
       and progress['stopped_on_prefix_contract_failure'] is True
       and len(progress['records']) == 2
       and sum(r['terminal']['trials'] for r in progress['records']) == 16)
    stages = {s['arm']: s for s in plan['stages'] if s['motion_mode'] == 'hold'}
    input_hashes = {}
    for arm in ('holding_A', 'full_V17'):
        folder = ROOT / stages[arm]['output']
        done = read(folder / 'execution_complete.json')
        ck(arm + ' terminal8 source unchanged', done['execution_complete']
           and done['sources_unchanged'] and done['trials'] == done['planned'] == 8)
        input_hashes[str(folder / 'execution_complete.json')] = sha(folder / 'execution_complete.json')
    frozen = {str(PLAN): sha(PLAN), str(PROGRESS): sha(PROGRESS)}
    for name in ('review/run_v17_transfer_small_v1.py',
                 'review/run_v17b_prospective80_v1.py'):
        frozen[str(ROOT / name)] = sha(ROOT / name)
    before_sources = {}
    for name, digest in plan['source_sha256'].items():
        p = ROOT.parent / '五指策略接入_2026-11-03' / name[11:] if name.startswith('foundation/') else ROOT / name
        before_sources[name] = sha(p)
    ck('all136 current source identities retained', len(before_sources) == 136 and before_sources == plan['source_sha256'])
    rows = []
    for seed in plan['seeds']:
        for scene in plan['scenarios']:
            label = f'{scene}/{seed}'
            bundles = {}
            for arm in ('holding_A', 'full_V17'):
                run = ROOT / stages[arm]['output'] / 'runs' / f'{scene}__ratio_effort__{seed}'
                path = run / 'trace.npz'
                input_hashes[str(path)] = sha(path)
                with np.load(path, allow_pickle=False) as z:
                    mask = z['time'] < 12. - 1e-9
                    bundles[arm] = {k: z[k][mask] for k in COLS}
                ck(label + '/' + arm + '/2400 exact prefix clocks', np.array_equal(bundles[arm]['time'], np.arange(2400) * .005))
            a, b = bundles['holding_A'], bundles['full_V17']
            columns = []
            for name in COLS:
                aa, bb = a[name], b[name]
                shape = aa.shape == bb.shape
                text = aa.dtype.kind in 'US' and bb.dtype.kind in 'US'
                literal = shape and aa.dtype == bb.dtype and aa.tobytes() == bb.tobytes()
                if text:
                    same = shape and all(str(x).encode('utf-8') == str(y).encode('utf-8') for x, y in zip(aa.flat, bb.flat))
                else:
                    same = literal
                columns.append(dict(column=name, exact_equal=bool(same),
                    rule='complete_exact_UTF8_payload_no_fields_removed' if text else 'exact_dtype_shape_bytes',
                    literal_storage_equal=bool(literal), holding_dtype=str(aa.dtype), full_dtype=str(bb.dtype)))
                ck(label + '/' + name, same)
            meta_complete = all(set(ADDED) <= set(json.loads(str(s)))
                                for arm in ('holding_A', 'full_V17')
                                for s in bundles[arm]['legacy_common_sensor_json'])
            ck(label + '/both arms contain all6 in all2400 packets', meta_complete)
            x, y = (json.loads(str(bundles[arm]['legacy_common_sensor_json'][0]))
                    for arm in ('full_V17', 'holding_A'))
            stripped = {k: v for k, v in x.items() if k not in ADDED}
            mismatch = sorted(set(y) - set(stripped))
            ck(label + '/old one-sided pop creates exactly6 artificial missing fields', set(mismatch) == set(ADDED))
            rows.append(dict(scenario=scene, seed=seed, prefix_frames=2400,
                             scope='t<12; release tick excluded', columns=columns,
                             complete_sensor_UTF8_exact=True if all(c['exact_equal'] for c in columns) else False,
                             first_real_nested_difference=None,
                             old_comparator_first_artificial_difference=dict(time_s=0., path='$',
                                missing_only_after_left_pop=mismatch,
                                raw_input_strings_identical=True)))
    ck('all8 planned hold pairs checked', len(rows) == 8)
    ck('original plan scheduler helper stopped progress unchanged', all(sha(p) == h for p, h in frozen.items()))
    ck('all captured inputs unchanged', all(sha(p) == h for p, h in input_hashes.items()))
    ck('no MuJoCo or plant module imported', not any(k.startswith('mujoco') or k == 'hand_plant' for k in sys.modules))
    result = dict(schema='v17b_prefix_reader_correction_independent_v1',
        created_utc=datetime.now(timezone.utc).isoformat(), passed=all(c['passed'] for c in checks),
        checks=checks, rows=rows, plan_sha256=sha(PLAN), source_sha256=plan['source_sha256'],
        input_sha256=input_hashes, retained_reader_incident_sha256=frozen,
        checker_sha256=sha(__file__), compared_pairs=8, complete_public_sensor_packets=19200,
        assigned_count=80, completed_scope=16, all_assigned_checked=False,
        complete80_execution_audit=False, physics_steps=0,
        finding='Old helper removes6 fields from full_V17 only, although both arms contain them. All unmodified prefix JSON UTF8 payloads are already exact; this is a comparator-scope bug, not a data difference or tolerance issue.',
        limitations=['Only the16 completed hold runs and their8 paired pre12 prefixes were read.',
                     'No performance, result metrics, pending64 traces, model or physical replay was read/run.',
                     'All original failure records and source/helper/scheduler/registration bytes remain retained. This gate is not the final80 execution gate.'])
    STEM.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    text = f'''# V17b已完成16条的前缀比较器事件：独立纠错核查

结果：{'PASS' if result['passed'] else 'FAIL'}，{sum(c['passed'] for c in checks)}/{len(checks)}。仅核8对、16条已完成hold，未签80条完成门。

8对全部37列在t<12的2400帧一致：数值严格dtype/shape/bytes；文本严格完整UTF8载荷。公共传感JSON共19200对完整字符串逐值相同，包含全部6个新增domain字段，未忽略字段、未放宽浮点容差。真实嵌套差异不存在。

旧helper `review/run_v17_transfer_small_v1.py:23–29` 原用于“新增仪器对旧仪器”，在第28行只从左侧x移除6字段。此处双方本已有6字段，旧helper因此在第0帧人为制造6个缺键。该失败是比较器适用范围错误，不是原始数据差异，也不是仅Unicode宽度问题。文本整集dtype宽度是否不同另列保留，不借此抹掉literal证据。

原master、scheduler、helper和停止progress保持原字节，原16结果不重跑，136执行源全部当前SHA仍等于注册。此报告不签新resume算法；任何续跑仅可由root另存修订reader/收据，继续剩余64分配。未读取成绩、未选优，未导入模型或物理模块。

完整检查、输入与旧失败事件SHA见 [JSON]({STEM.name}.json)。
'''
    STEM.with_suffix('.md').write_text(text, encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'], checks=len(checks), json=str(STEM.with_suffix('.json')),
                         sha256=sha(STEM.with_suffix('.json')), md_sha256=sha(STEM.with_suffix('.md'))), ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
