"""Offline approach peak/event alignment; public clocks kept distinct from raw."""
from pathlib import Path
import gzip,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    source=ROOT/'review/v17_release_mechanisms_full15_v1.json';prior=read(source);rows=[]
    for item in prior['records']:
        if item['role']!='contact_V17':continue
        p=Path(item['run']);m=read(p/'metadata.json')
        with np.load(p/'trace.npz',allow_pickle=False) as z:
            a={k:z[k] for k in ('time','observation_json','executed_reference21','substep_time','substep_duration','substep_skin_normal','substep_palm','lift')}
        with np.load(p/'delivery_intervals.npz',allow_pickle=False) as z:
            l={k:z[k] for k in ('interval_start_s','interval_end_s','target_receiver_top_normal_N','receiver_any_actor_top_normal_N','target_hand_normal_N','object_free_qpos7_end','object_free_qvel6_end')}
        with gzip.open(p/'release_decisions.jsonl.gz','rt',encoding='utf-8') as f:ds=[json.loads(line) for line in f]
        times=a['time'];selected=np.flatnonzero((times>=12-1e-9)&(times<=13+1e-9));obs={int(i):json.loads(str(a['observation_json'][i])) for i in selected}
        phase_contact=next(d['time_s'] for d in ds if d['phase']=='CONTACT_UNLOADING')
        peaks={}
        for label,stop in [('APPROACH',phase_contact),('RELEASE',16.)]:
            mask=(a['substep_time']>=12-1e-9)&(a['substep_time']<stop-1e-9);ids=np.flatnonzero(mask)
            flat=int(np.argmax(a['substep_skin_normal'][mask]));j,k=np.unravel_index(flat,a['substep_skin_normal'][mask].shape);i=int(ids[j])
            start=float(a['substep_time'][i]);end=float(start+a['substep_duration'][i]);li=int(np.argmin(abs(l['interval_end_s']-end)))
            near=int(np.searchsorted(times,end+1e-9)-1);o=obs[near];r=o['receiver_surface'];before=[d for d in ds if d['time_s']<=start+1e-9][-1]
            peaks[label]=dict(finger_index=int(k),force_N=float(a['substep_skin_normal'][i,k]),interval_start_s=start,interval_end_s=end,
                preceding_public_time_s=float(times[near]),public_acquisition_time_s=o['sample_time'],
                receiver_acquisition_time_s=r['acquisition_time_s'],receiver_normal_load_N=r['normal_load_N'],receiver_valid=r['valid'],
                public_normal5_N=o['normal'],public_hand_acquisition_s=o['sample_time'],
                actual_lift_encoder_m=float(a['lift'][near]),actual_lift_velocity_m_s=o['lift_velocity_m_s'],
                committed_lift_reference_post_action_m=float(a['executed_reference21'][near,18]),
                actual_minus_reference_lift_gap_m=float(a['lift'][near]-a['executed_reference21'][near,18]),
                preceding_decision_time_s=before['time_s'],preceding_decision_phase=before['phase'],
                preceding_plan_delta_lift_m=before['proposed_reference_increment21'][18],
                target_top_cached_force_N=float(l['target_receiver_top_normal_N'][li]),
                receiver_any_actor_top_cached_force_N=float(l['receiver_any_actor_top_normal_N'][li]),
                target_hand_cached_force_N=float(l['target_hand_normal_N'][li]),
                object_q7_end=l['object_free_qpos7_end'][li].tolist(),object_velocity6_end=l['object_free_qvel6_end'][li].tolist(),
                public_vs_cached_force_different_phases_not_direct_sample_equality=True)
        hits=np.flatnonzero((l['interval_end_s']>12+1e-9)&(l['target_receiver_top_normal_N']>0));ft=int(hits[0]);top_end=float(l['interval_end_s'][ft])
        reactive=[]
        for i in selected:
            o=obs[int(i)];r=o['receiver_surface'];acq=r['acquisition_time_s']
            eligible=r['valid'] and acq is not None and acq>=12-1e-9
            compression=eligible and r['normal_load_N']>r['uncertainty6'][2]
            if compression:reactive.append((int(i),o))
        fi,first=reactive[0];fr=first['receiver_surface']
        timing=dict(first_target_top_cached_interval_end_s=top_end,first_public_compression_s=float(times[fi]),first_public_compression_acquisition_s=fr['acquisition_time_s'],
            first_public_compression_load_N=fr['normal_load_N'],CONTACT_observation_time_s=phase_contact,
            first_target_top_to_first_public_compression_s=float(times[fi]-top_end),first_target_top_to_CONTACT_s=float(phase_contact-top_end),
            declared_reference_descent_during_latency_m=float(.18*(phase_contact-top_end)),
            latency_reference_displacement_role='rate times elapsed diagnostic only; actual mechanical motion separately logged')
        window=(times>=top_end-.025)&(times<=phase_contact+.025)
        public=[]
        for i in np.flatnonzero(window):
            o=obs[int(i)];r=o['receiver_surface']
            public.append(dict(time_s=float(times[i]),receiver_acquisition_s=r['acquisition_time_s'],receiver_valid=r['valid'],R_N=r['normal_load_N'],R_uncertainty_N=r['uncertainty6'][2],
                hand_acquisition_s=o['sample_time'],normal5_N=o['normal'],aux_pressure_N=o['auxiliary_pressure'],
                lift_m=float(a['lift'][i]),lift_velocity_m_s=o['lift_velocity_m_s'],post_action_lift_ref_m=float(a['executed_reference21'][i,18]),
                reference_gap_m=float(a['lift'][i]-a['executed_reference21'][i,18])))
        rows.append(dict(context=item['context'],run=str(p),inputs_sha256={n:sha(p/n) for n in ('trace.npz','delivery_intervals.npz','release_decisions.jsonl.gz')},
            event_timing=timing,peaks=peaks,public_near_contact=public,
            original_limits5_N=m['limits'],original_budget_N=m['budget']))
    out=ROOT/'review/v17_approach_event_alignment_v1.json'
    if out.exists():raise FileExistsError(out)
    doc=dict(schema='v17_approach_event_alignment_offline_v1',complete=len(rows)==3,records=rows,
        source_full15_sha256=sha(source),reader_sha256=sha(__file__),physics_steps=0,
        scope='posthoc event alignment, target identity only offline ledger, no controller/model execution, causal hypotheses require matched interventions')
    out.write_text(json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for r in rows:print(json.dumps(dict(context=r['context'],timing=r['event_timing'],peaks=r['peaks']),ensure_ascii=False))
    print(json.dumps(dict(out=str(out),sha256=sha(out))))
if __name__=='__main__':main()
