"""Create a separate five-arm archive viewer without touching older viewers."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'v15A_replay_v1';NEW=ROOT/'v17b_replay_v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def replace(s,a,b):
    if a not in s:raise ValueError('Expected parent text missing: '+a[:80])
    return s.replace(a,b)


def main():
    if NEW.exists():raise FileExistsError(NEW)
    pin=json.loads((ROOT/'review/v17b_prospective80_plan_v1.json').read_text(encoding='utf-8'))
    plan_sha=sha(ROOT/'review/v17b_prospective80_plan_v1.json')
    parent={name:sha(OLD/name) for name in ('reader.py','rendering.py','server.py','web/index.html')}
    s=(OLD/'reader.py').read_text(encoding='utf-8')
    changes={
        'V15A64':'V17b80','all four registered batches':'all six registered batches',
        'review/v15A_prospective_multiscene_plan_v2.json':'review/v17b_prospective80_plan_v1.json',
        '87f802635f276e5f9a27ce2cd8548dc2cd02a8b1ef262206c5d165953a84b074':plan_sha,
        "METHODS={'candidate_A':'多任务协调 A','baseline_ratio':'力比 PI 基线',":"METHODS={'full_V17':'完整多任务 V17','holding_A':'仅持握改进 A','baseline_ratio':'力比 PI 基线',",
        "SEEDS={619731:'条件 A',619732:'条件 B'}":"SEEDS={619741:'新条件 A',619742:'新条件 B'}",
        "if arm=='candidate_A' and method=='ratio_effort':return 'candidate_A'":"if arm=='full_V17' and method=='ratio_effort':return 'full_V17'\n    if arm=='holding_A' and method=='ratio_effort':return 'holding_A'",
        "if arm=='baseline':":"if arm=='baselines':",
        'cache_records=8':'cache_records=10','not 4<=cache_records<=8':'not 5<=cache_records<=10',
        'Cache must contain4..8 records':'Cache must contain5..10 records',
        "len(self.registration['source_sha256'])!=114":"len(self.registration['source_sha256'])!=136",
        'frozen source114':'frozen source136','Registered source114 differs':'Registered source136 differs',
        'registered V15A v2 plan':'registered V17b80 plan',
        'len(records)!=64':'len(records)!=80','Expected16 contexts×4 distinct roles':'Expected16 contexts×5 distinct roles',
        "metadata.get('receiver_policy')!='blind'":"metadata.get('receiver_policy')!=r['stage']['receiver_policy']",
        '整批64条尚未完成':'整批80条尚未完成',
        "self._diagnostic=None":"self._diagnostic=None;self._placement_module=None",
        "_v15A_replay_frozen_diagnostic":"_v17b_replay_frozen_diagnostic"
    }
    for a,b in changes.items():s=replace(s,a,b)
    needle='    def load_trace(self,scenario,motion,seed,method):'
    addition='''    def _placement(self):
        if self._placement_module is None:
            p=self.registration['auxiliary_metrics'];path=ROOT/p['reader']
            if sha(path)!=p['reader_sha256'] or sha(ROOT/p['contract'])!=p['contract_sha256']:
                raise ValueError('Registered placement calculator/contract changed')
            spec=importlib.util.spec_from_file_location('_v17b_replay_auxiliary_v3',path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            self._placement_module=module
        return self._placement_module

'''
    s=replace(s,needle,addition+needle)
    needle="            data.update(metadata=metadata,batch=r['batch'],folder=r['folder'],trace_sha256=expected,"
    addition="""            placement=None
            paux=self._placement()
            if all(key in ledger for key in paux.C['input_fields']):
                pa=paux.calculate(ledger,release_at_s=metadata.get('mechanical_release_at',metadata.get('release_at')),
                    episode_end_s=metadata['duration'],physical_period_s=metadata['physics_timestep'],original_result=r['result'])
                placement={key:pa[key] for key in ('status','official_first_stable_receiver',
                    'official_first_stable_by_release_deadline','official_final_stable_suffix',
                    'no_stability_reason','ever_floor_contact','stable_after_previous_floor_event',
                    'first_receiver_impact20ms','receiver_event_provenance')}
            data['placement']=placement
"""
    s=replace(s,needle,addition+needle)
    needle="                reason=r.get('reason','整批80条尚未完成，暂不显示部分对照'))"
    s=replace(s,needle,"                placement=d['placement'] if available else None,\n"+needle)
    NEW.mkdir();(NEW/'web').mkdir();(NEW/'__init__.py').write_text('',encoding='utf-8')
    (NEW/'reader.py').write_text(s,encoding='utf-8')
    s=(OLD/'rendering.py').read_text(encoding='utf-8').replace('v15A-recorded-render-owner','v17b-recorded-render-owner')
    (NEW/'rendering.py').write_text(s,encoding='utf-8')
    s=(OLD/'server.py').read_text(encoding='utf-8')
    a=s.index('def check_audit(path):');b=s.index('\nclass Handler',a)
    s=s[:a]+'''def check_audit(path):
    report=read(path)
    if report.get('schema')!='v17b_prospective80_execution_independent_v1' or report.get('passed') is not True:
        raise ValueError('Full80 independent execution audit must pass before serving')
    if report.get('plan_sha256')!=PLAN_SHA or report.get('assigned_count')!=80 or report.get('all_assigned_checked') is not True:
        raise ValueError('Audit/plan/full assignment differs')
'''+s[b:]
    for a,b in {
        'review/v15A_prospective64_independent_v2.json':'review/v17b_prospective80_execution_independent_v1.json',
        '四方法':'五方法','roles=4':'roles=5','default=8893':'default=8894',
        '(8878,8880,8892)':'(8878,8880,8892,8893)','default8893':'default8894','All64':'All80'}.items():s=replace(s,a,b)
    (NEW/'server.py').write_text(s,encoding='utf-8')
    s=(OLD/'web/index.html').read_text(encoding='utf-8')
    for a,b in {'四方法':'五方法','四条真实记录':'五条真实记录','64条记录':'80条记录',
        'value="619731"':'value="619741"','value="619732"':'value="619742"',
        'repeat(4,minmax(0,1fr))':'repeat(5,minmax(0,1fr))',
        "const METHODS=[['candidate_A','多任务协调 A','协调候选'],":"const METHODS=[['full_V17','完整多任务 V17','接触转移'],['holding_A','仅持握改进 A','机制对照'],",
        '.card[data-role=baseline_ratio]':'.card[data-role=holding_A]{--color:#6a9b83}.card[data-role=baseline_ratio]',
        '@media(max-width:1200px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}':'@media(max-width:1400px){.cards{grid-template-columns:repeat(3,minmax(0,1fr))}',
        '@media(max-width:680px)':'@media(max-width:1000px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:680px)',
        '多任务协调 A 使用保守力比偏好与共同任务协调':'完整V17与仅持握A共用持握任务，完整V17另含接触转移释放',
        '<span class="peak">全程峰值 / 超限剂量：不可评</span>':'<span class="peak">全程峰值 / 超限剂量：不可评</span><span class="settle">首次低运动驻留：不可评</span><span class="final-settle">末段低运动驻留：不可评</span><span class="impact">触台20ms峰 / 冲量：不可评</span>'}.items():s=replace(s,a,b)
    needle="fmt(m.excess_Ns,5)+' N·s'}$('help')"
    addition="""fmt(m.excess_Ns,5)+' N·s';const pa=r.placement;const complete=pa?.status==='computed_complete';const first=pa?.official_first_stable_receiver;const suffix=pa?.official_final_stable_suffix;el.querySelector('.settle').textContent='首次低运动驻留：'+(!complete?'不可评':first?fmt(first.latency_s)+' s（释放后）':'未达到');el.querySelector('.final-settle').textContent='末段低运动驻留：'+(!complete?'不可评':suffix&&!suffix.right_censored?fmt(suffix.duration_s)+' s':suffix?'末段不足0.2秒':'未达到');const impact=pa?.first_receiver_impact20ms;const f=impact?.contact?.receiver_top;el.querySelector('.impact').textContent='触台20ms峰 / 冲量：'+(impact?.available?fmt(f?.peak_N)+' N / '+fmt(f?.impulse_Ns,5)+' N·s':'无可评新接触事件')} $('help')"""
    s=replace(s,needle,addition)
    (NEW/'web/index.html').write_text(s,encoding='utf-8')
    report=dict(schema='v17b_archive_viewer_generation_v1',plan_sha256=plan_sha,parent_sources_sha256=parent,
        generated_sha256={str(p.relative_to(NEW)):sha(p) for p in NEW.rglob('*') if p.is_file()},
        algorithms_or_physics_modified=False,rendering_started=False,server_started=False,
        old_viewers_unchanged=all(sha(OLD/n)==h for n,h in parent.items()),all80_audit_required_before_serve=True)
    (NEW/'generation_v1.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(created=True,old_unchanged=report['old_viewers_unchanged'],files=len(report['generated_sha256']),plan_sha256=plan_sha)))


if __name__=='__main__':main()
