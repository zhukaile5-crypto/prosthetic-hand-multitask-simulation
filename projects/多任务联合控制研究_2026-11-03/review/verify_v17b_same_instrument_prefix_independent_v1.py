"""Finite synthetic gate of the replacement reader; no archive re-reading."""
from pathlib import Path
from unittest.mock import patch
import hashlib
import importlib.util
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT.parent / '五指策略接入_2026-11-03'))
sys.dont_write_bytecode = True
HELPER = ROOT / 'review/v17b_same_instrument_prefix_v1.py'
PRIOR = ROOT / 'review/v17b_prefix_reader_correction_independent_v1.json'
STEM = ROOT / 'review/v17b_same_instrument_prefix_independent_v1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Archive:
    def __init__(self, values): self.values = values
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def __contains__(self, key): return key in self.values
    def __getitem__(self, key): return self.values[key]


def main():
    if any(STEM.with_suffix(s).exists() for s in ('.json', '.md')):
        raise FileExistsError(STEM)
    before = sha(HELPER)
    spec = importlib.util.spec_from_file_location('new_literal_prefix_reader_independent', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    checks, cases = [], []
    def ck(name, ok): checks.append(dict(name=name, passed=bool(ok)))
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    ck('bind original337 PASS8pairs19200 exact packets', prior['passed']
       and len(prior['checks']) == 337 and prior['compared_pairs'] == 8
       and prior['complete_public_sensor_packets'] == 19200
       and sha(PRIOR) == 'c5f5e5dbe795b7ed74857260a4ef2fd9b1039e9032872ac34299032a2638f7e8')
    columns = tuple(helper.PHYSICAL_COLUMNS) + ('raw_patches_json',)
    ck('all37 original columns included', len(columns) == len(set(columns)) == 37)
    fields = dict(slip_valid5=[False] * 5, slip_domain_acquisition_time_s=0.,
        slip_domain_time_s=0., slip_domain_role='VALIDITY_OF_EXISTING_INSTRUMENT_SLIP_CHANNEL',
        slip_domain_available=True, slip_domain_reason=None, payload='a')
    def fixture():
        data = {k: np.array([1., 2., 3.]) for k in columns}
        data['time'] = np.array([0., .005, 12.])
        for k in ('phase', 'command', 'receiver_measurement_json', 'task_request_json',
                  'environment_json', 'raw_patches_json'):
            data[k] = np.array(['same', 'next', 'post'])
        data['legacy_common_sensor_json'] = np.array([json.dumps(fields)] * 3)
        return data
    def call(name, a, b, wanted):
        def load(p, **kwargs): return Archive(a if Path(p).parent.name == 'full' else b)
        with patch.object(helper.np, 'load', load), patch.object(helper, 'sha', lambda p: 'synthetic-not-a-file'):
            result = helper.compare(Path('full'), Path('holding'))
        cases.append(dict(name=name, expected_passed=wanted, result=result))
        ck(name, result['passed'] is wanted)
        return result
    call('identical complete input passes', fixture(), fixture(), True)
    a, b = fixture(), fixture()
    s = a['legacy_common_sensor_json'].tolist()
    changed = json.loads(s[1]); changed['payload'] = 'b'; s[1] = json.dumps(changed)
    a['legacy_common_sensor_json'] = np.array(s)
    r = call('one JSON character outside6metadata fails', a, b, False)
    ck('character failure attributed completeJSON while6metadata still same',
       not r['checks']['legacy_common_sensor_json']['equal'] and r['additional_domain_metadata_exact'])
    a, b = fixture(), fixture()
    a['observed_normal'][1] = np.nextafter(a['observed_normal'][1], np.inf)
    r = call('one scalar ULP change fails', a, b, False)
    ck('numeric failure exact, not a tolerance', not r['checks']['observed_normal']['equal'])
    a, b = fixture(), fixture()
    a['command'] = np.array(['same', 'next', 'post' * 100])
    ck('padding fixture has genuinely different dtype and prefix bytes',
       a['command'].dtype != b['command'].dtype and a['command'][:2].tobytes() != b['command'][:2].tobytes())
    call('whole episode Unicode width differs but prefix UTF8 exact passes', a, b, True)
    a, b = fixture(), fixture()
    missing = dict(fields); missing.pop('slip_valid5')
    for data in (a, b): data['legacy_common_sensor_json'] = np.array([json.dumps(missing)] * 3)
    r = call('same missing metadata on both sides fails', a, b, False)
    ck('missingmetadata failure independent of raw payload equality',
       r['checks']['legacy_common_sensor_json']['equal'] and not r['additional_domain_metadata_exact'])
    ck('replacement reader source unchanged', sha(HELPER) == before)
    result = dict(schema='v17b_same_instrument_prefix_independent_v1',
        passed=all(c['passed'] for c in checks), checks=checks, cases=cases,
        helper_sha256=before, original337_artifact_sha256=sha(PRIOR),
        checker_sha256=sha(__file__), real_archive_reads=0, model_constructions=0,
        physics_steps=0, final80_execution_gate=False,
        scope='Static all37/full-JSON match337 criteria plus five actual-helper synthetic cases; not new physical evidence or a resume scheduler audit.',
        caveat='This comparator assumes outer audit verifies complete3201 clocks/2400 prefix frames; it is not a full-run completeness validator.')
    STEM.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    STEM.with_suffix('.md').write_text(f'''# V17b同仪器前缀比较器：独立有限门

{'PASS' if result['passed'] else 'FAIL'}，{sum(c['passed'] for c in checks)}/{len(checks)}。绑定helper SHA `{before}` 与既有337项实际8对纠错证据SHA `{sha(PRIOR)}`。

直接调用新helper的五个内存fixture：完全一致通过；JSON非metadata单字符差异拒绝；数值1ULP差异拒绝；仅episode后段导致Unicode宽度变化但前缀UTF8相同通过；双方同缺一个metadata字段仍拒绝。完整37列/完整JSON均比较，无字段投影，无数值容差。

本次没有重解压8对、读取成绩、创建模型或积分。完整3201时钟、2400前缀帧数量由外层执行独审负责，本helper不是完整轨迹检查器。此证据不签80完成或resume调度器。

[完整JSON]({STEM.name}.json)。
''', encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'], checks=len(checks), helper_sha256=before,
                         json_sha256=sha(STEM.with_suffix('.json')),
                         md_sha256=sha(STEM.with_suffix('.md'))), ensure_ascii=False))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
