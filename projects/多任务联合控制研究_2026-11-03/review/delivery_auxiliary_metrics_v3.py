"""OFFLINE-only sampled receiver stability and contact-conversion diagnostics.

No control/model/simulator imports; all identity/state is evaluation-sidecar
input. Original results are preserved verbatim, never re-scored or filtered.
"""
from pathlib import Path
from copy import deepcopy
import json
import numpy as np

CONTRACT_PATH=Path(__file__).with_name('delivery_auxiliary_contract_v3.json')
C=json.loads(CONTRACT_PATH.read_text(encoding='utf-8'))
TH=C['thresholds'];EPS=TH['clock_tolerance_s']
CATS=('receiver_top','receiver_otherface','floor','other_nonhand_world','hand','other_nonworld')

def _plain(x):
    if isinstance(x,np.ndarray):return _plain(x.tolist())
    if isinstance(x,np.generic):return _plain(x.item())
    if isinstance(x,dict):return {k:_plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [_plain(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def _coverage(start,end,a,b,valid,period):
    selected=(end>a+1e-12)&(start<b-1e-12)
    duration=np.maximum(0.,np.minimum(end,b)-np.maximum(start,a));cursor=a;gaps=[];overlaps=[]
    ids=np.flatnonzero(selected)
    for i in ids:
        lo=max(a,start[i]);hi=min(b,end[i])
        if lo>cursor+EPS:gaps.append([float(cursor),float(lo)])
        if lo<cursor-EPS:overlaps.append([float(lo),float(cursor)])
        cursor=max(cursor,hi)
    if cursor<b-EPS:gaps.append([float(cursor),float(b)])
    oversized=ids[(end[ids]-start[ids])>period+EPS]
    covered=not gaps and not overlaps and not len(oversized) and len(ids)>0
    return dict(complete=bool(covered),all_valid=bool(covered and valid[ids].all()),
        covered_duration_s=float(duration.sum()),gaps=gaps,overlaps=overlaps,
        oversized_interval_count=len(oversized)),duration,ids

def _dwell(ends,mask,continuity,window,dwell):
    a,b=window;ids=np.flatnonzero((ends>=a-EPS)&(ends<=b+EPS));since=None;previous=None;first=None;episodes=[];segment_confirmation=None
    for i in ids:
        t=float(ends[i]);t=a if abs(t-a)<=EPS else b if abs(t-b)<=EPS else t
        good=bool(mask[i])
        if previous is not None and not continuity[i]:
            if since is not None:episodes.append(dict(start_s=since,end_s=float(ends[previous]),broken_by='missing_interval'))
            since=None;segment_confirmation=None
        if good:
            if since is None:since=max(a,t);segment_confirmation=None
            if t-since>=dwell-EPS and segment_confirmation is None:segment_confirmation=t
            if first is None and segment_confirmation is not None:first=dict(start_s=since,confirmation_s=segment_confirmation,latency_s=segment_confirmation-a)
        elif since is not None:
            episodes.append(dict(start_s=since,end_s=float(ends[previous]) if previous is not None else t,broken_by='condition_or_validity'))
            since=None;segment_confirmation=None
        previous=i
    suffix=None;censored=False
    if since is not None and previous is not None and abs(float(ends[previous])-b)<=EPS:
        length=b-since;censored=length<dwell-EPS
        suffix=dict(start_s=since,confirmation_s=segment_confirmation if not censored else None,
                    end_s=b,duration_s=length,right_censored=censored)
    return dict(first=first,final_suffix=suffix,final_suffix_insufficient_dwell=censored,qualifying_episodes=episodes)

def calculate(arrays,*,release_at_s,episode_end_s=16.,physical_period_s=.0005,original_result=None):
    """No missing record is substituted with zero or a successful outcome."""
    a=float(release_at_s);b=float(episode_end_s);period=float(physical_period_s)
    if not np.isfinite([a,b,period]).all() or not 0<=a<b or period<=0:raise ValueError('Declared finite positive release window/period required')
    start=np.asarray(arrays['interval_start_s'],float);end=np.asarray(arrays['interval_end_s'],float);n=len(start)
    if start.shape!=(n,) or end.shape!=(n,) or not np.isfinite(np.r_[start,end]).all() or np.any(end<=start) or np.any(np.diff(end)<=0):raise ValueError('Finite strictly ordered positive intervals required')
    flags={k:np.asarray(arrays[k]) for k in ('contact_valid','object_state_valid')}
    if any(v.shape!=(n,) or v.dtype!=bool for v in flags.values()):raise ValueError('Explicit complete validity masks required')
    q=np.asarray(arrays['object_free_qpos7_end'],float);v=np.asarray(arrays['object_free_qvel6_end'],float)
    if q.shape!=(n,7) or v.shape!=(n,6):raise ValueError('Free-joint END state shapes required')
    force={k:np.asarray(arrays['target_'+k+'_normal_N'],float) for k in CATS}
    if any(x.shape!=(n,) for x in force.values()):raise ValueError('Every disjoint target contact category required')
    fmat=np.array(list(force.values())).T
    cv=flags['contact_valid']&np.isfinite(fmat).all(1)&(fmat>=0).all(1)
    sv=flags['object_state_valid']&np.isfinite(q).all(1)&np.isfinite(v).all(1)
    sv &= abs(np.linalg.norm(q[:,3:],axis=1)-1)<=1e-5
    lin=np.linalg.norm(v[:,:3],axis=1);ang=np.linalg.norm(v[:,3:],axis=1)
    continuity=np.ones(n,bool)
    if n:continuity[0]=False
    if n>1:continuity[1:]=(abs(start[1:]-end[:-1])<=EPS)&((end[1:]-end[:-1])<=period+EPS)
    cov,duration,ids=_coverage(start,end,a,b,cv&sv,period)
    ccov,_,_=_coverage(start,end,a,b,cv,period)
    rawlow=cv&(force['hand']<=TH['all_target_hand_contact_clearance_N'])
    stable=(cv&sv&(force['receiver_top']>0)&rawlow&(lin<=TH['absolute_free_joint_linear_speed_m_s'])
        &(ang<=TH['absolute_free_joint_angular_speed_rad_s']))
    for category in ('receiver_otherface','floor','other_nonhand_world','other_nonworld'):stable &= force[category]==0
    sd=_dwell(end,stable,continuity,(a,b),TH['stable_dwell_s']);hd=_dwell(end,rawlow,continuity,(a,b),TH['raw_hand_clearance_dwell_s'])
    known=cov['all_valid'];known_contact=ccov['all_valid'];deadline=min(b,a+TH['original_release_deadline_s'])
    first=sd['first'] if known else None
    first_deadline=first if first is not None and first['confirmation_s']<=deadline+EPS else None
    floor_ids=np.flatnonzero((duration>0)&cv&(force['floor']>0));first_floor=float(max(a,start[floor_ids[0]])) if len(floor_ids) else None
    top_ids=np.flatnonzero((duration>0)&cv&(force['receiver_top']>0));first_top=float(max(a,start[top_ids[0]])) if len(top_ids) else None
    everfloor=True if len(floor_ids) else False if known_contact else None
    # Use only an actual command-time cached record, not a future interval.
    command_hits=np.flatnonzero(abs(end-a)<=EPS)
    def contact_provenance(category):
        initial=None
        if len(command_hits)==1 and cv[command_hits[0]]:
            initial=bool(force[category][command_hits[0]]>0)
        new=[]
        for i in np.flatnonzero((duration>0)&cv&(force[category]>0)):
            if i>0 and cv[i-1] and force[category][i-1]==0 and abs(start[i]-end[i-1])<=EPS and start[i]>=a-EPS:
                new.append(float(max(a,start[i])))
        observed=np.flatnonzero((duration>0)&cv&(force[category]>0))
        first_i=int(observed[0]) if len(observed) else None
        first_is_new=bool(first_i is not None and first_i>0 and cv[first_i-1] and force[category][first_i-1]==0
            and abs(start[first_i]-end[first_i-1])<=EPS and start[first_i]>=a-EPS)
        onset_status=('NO_OBSERVED_POSITIVE_CONTACT' if first_i is None else
            'PREEXISTING_AT_RELEASE' if initial is True else
            'KNOWN_ZERO_TO_POSITIVE_TRANSITION' if first_is_new else
            'UNKNOWN_PRIOR_CONTACT_OR_ONSET')
        return dict(contact_at_command=initial,
            observed_positive_contact=bool(len(observed)),first_positive_onset_status=onset_status,
            first_observed_positive_interval_s=None if first_i is None else [float(max(a,start[first_i])),float(min(b,end[first_i]))],
            initial_contact_role='PREEXISTING_AT_RELEASE' if initial is True else 'ABSENT_AT_RELEASE' if initial is False else 'UNKNOWN_COMMAND_CONTACT',
            first_observed_new_transition_s=new[0] if new else None,
            observed_new_transition_count=len(new),
            no_new_impact_inferred_from_preexisting_or_unknown_load=True)
    receiver_origin=contact_provenance('receiver_top');floor_origin=contact_provenance('floor')
    terminal=None
    hits=np.flatnonzero(abs(end-b)<=EPS)
    if len(hits)==1:
        j=int(hits[0]);stateknown=bool(sv[j]);contactknown=bool(cv[j]);lowmotion=bool(lin[j]<=TH['absolute_free_joint_linear_speed_m_s'] and ang[j]<=TH['absolute_free_joint_angular_speed_rad_s']) if stateknown else None
        terminal=dict(time_s=float(end[j]),object_state_available=stateknown,contact_available=contactknown,
            position_m=q[j,:3].tolist() if stateknown else None,linear_speed_m_s=float(lin[j]) if stateknown else None,
            angular_speed_rad_s=float(ang[j]) if stateknown else None,
            target_receiver_top_contact=bool(force['receiver_top'][j]>0) if contactknown else None,
            target_floor_contact=bool(force['floor'][j]>0) if contactknown else None,
            target_hand_clear=bool(rawlow[j]) if contactknown else None,low_absolute_motion=lowmotion,
            moving_receiver_contact=bool(force['receiver_top'][j]>0 and not lowmotion) if contactknown and lowmotion is not None else None,
            rolling_identified=False,rolling_role='Velocity norms alone do not distinguish rolling/sliding/rocking',
            complete_object_containment_on_receiver_known=False)
    def impact(onset,width):
        if onset is None:return None
        stop=onset+width;limited=min(b,stop);co,dt,ii=_coverage(start,end,onset,limited,cv,period)
        complete=co['all_valid'] and stop<=b+EPS
        values={}
        for category in ('receiver_top','hand','floor'):
            valid_ids=ii[cv[ii]]
            peak=float(np.max(force[category][valid_ids])) if len(valid_ids) else None
            integral=float(np.sum(force[category][valid_ids]*dt[valid_ids])) if len(valid_ids) else None
            values[category]=dict(peak_N=peak if complete else None,impulse_Ns=integral if complete else None,
                observed_partial_peak_N=peak,observed_partial_impulse_Ns=integral)
        # State is sampled at END, never pull a later state into this window.
        state_ids=np.flatnonzero((end>=onset-EPS)&(end<=limited+EPS)&sv)
        return dict(start_s=onset,end_requested_s=stop,available=complete,right_censored=stop>b+EPS,
            coverage=co,contact=values,observed_END_linear_speed_max_m_s=float(np.max(lin[state_ids])) if len(state_ids) else None,
            observed_END_angular_speed_max_rad_s=float(np.max(ang[state_ids])) if len(state_ids) else None,
            no_state_interpolation_or_future_sample=True)
    if first is not None:
        after=(end>first['confirmation_s']+EPS)&(duration>0)
        bad=after&cv&sv&~stable;entries=bad&~np.r_[False,bad[:-1]]
        loss=dict(known_condition_loss_episodes=int(entries.sum()),known_not_stable_duration_s=float(duration[bad].sum()),
                  unknown_duration_s=float(duration[after&~(cv&sv)].sum()))
    else:loss=None
    return _plain(dict(schema='delivery_auxiliary_metrics_v3',status='computed_complete' if known else 'incomplete_or_invalid_unknown',
        original_result=deepcopy(original_result),window_s=[a,b],physical_period_s=period,
        coverage=cov,contact_coverage=ccov,official_first_stable_receiver=first,official_first_stable_by_release_deadline=first_deadline,
        original_release_deadline_at_s=deadline,official_final_stable_suffix=sd['final_suffix'] if known else None,
        observed_stable_candidates=sd,observed_raw_hand_clearance=hd,
        no_stability_reason=(None if first else 'ledger_incomplete_or_unknown' if not known else 'raw_hand_never_cleared' if hd['first'] is None else 'no_qualified_receiver_low_motion_dwell'),
        loss_after_first=loss,ever_floor_contact=everfloor,observed_first_floor_contact_s=first_floor,
        stable_after_previous_floor_event=(first_floor is not None and first is not None and first_floor<first['confirmation_s']) if known_contact and first is not None else None,
        terminal=terminal,first_receiver_contact_s=first_top,receiver_event_provenance=receiver_origin,floor_event_provenance=floor_origin,
        first_receiver_load_window20ms=impact(first_top,.02),first_receiver_load_window100ms=impact(first_top,.1),
        first_receiver_impact20ms=impact(receiver_origin['first_observed_new_transition_s'],.02),first_receiver_impact100ms=impact(receiver_origin['first_observed_new_transition_s'],.1),
        first_floor_load_window100ms=impact(first_floor,.1),first_floor_impact100ms=impact(floor_origin['first_observed_new_transition_s'],.1),
        original_release_physical_exposure=(original_result or {}).get('research_metrics',{}).get('release',{}).get('physical_exposure'),
        existing_support_not_attributed_to_release=True,
        original_metric_or_success_redefined=False,physical_or_permanent_placement_certified=False,
        no_filtered_force_or_weighted_total_score=True,observability_scope='Privileged offline ledger only, never a public controller measurement'))
