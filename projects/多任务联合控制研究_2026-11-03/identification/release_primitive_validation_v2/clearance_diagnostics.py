"""Additional descriptive first/final-clear diagnostics, never replace V3."""
import numpy as np


def clear_series(time_s, contact_N, release_s, threshold_N, dwell_s=.2, deadline_s=None):
    t, f = np.asarray(time_s, float), np.asarray(contact_N, float)
    if t.ndim!=1 or f.shape!=t.shape or len(t)<2 or not np.isfinite(np.r_[t,f]).all():
        raise ValueError('Finite same-length pressure clock series required')
    delta=np.diff(t)
    if np.any(delta<=0) or np.max(abs(delta-delta[0]))>1e-8:
        raise ValueError('Continuous fixed-step clock required; missing windows cannot be clear')
    ids=np.flatnonzero(t>=release_s-1e-9)
    if not len(ids):
        return dict(available=False,censored=True,reason='no_release_window')
    clear=f<=threshold_N
    first_start=first_confirmation=None
    start=None
    for i in ids:
        if clear[i]:
            if start is None:start=int(i)
            if t[i]-t[start]>=dwell_s-1e-9:
                first_start=float(t[start]);first_confirmation=float(t[i]);break
        else:start=None
    episodes=[]
    if first_confirmation is not None:
        onset=None
        for i in np.flatnonzero(t>first_confirmation+1e-9):
            if not clear[i] and onset is None:onset=int(i)
            if clear[i] and onset is not None:
                episodes.append(dict(start_s=float(t[onset]),observed_exit_s=float(t[i]),
                    duration_s=float(t[i]-t[onset]),censored=False));onset=None
        if onset is not None:
            episodes.append(dict(start_s=float(t[onset]),observed_exit_s=None,
                duration_s=float(t[-1]-t[onset]),censored=True))
    contact=np.flatnonzero((t>=release_s-1e-9)&~clear)
    last_exit=None
    if len(contact) and contact[-1]<len(t)-1:
        last_exit=float(t[contact[-1]+1])
    trailing=None
    if clear[-1]:
        j=len(t)-1
        while j>ids[0] and clear[j-1]:j-=1
        trailing=int(j)
    final_start=float(t[trailing]) if trailing is not None else None
    final_confirm=(final_start+dwell_s if final_start is not None and
                   t[-1]-final_start>=dwell_s-1e-9 else None)
    if final_confirm is not None:
        # Report an actually observed confirmation, not an invented sub-grid time.
        k=int(np.searchsorted(t,final_confirm-1e-9));final_confirm=float(t[k])
    deadline=float(t[-1] if deadline_s is None else min(deadline_s,t[-1]))
    first_by_deadline=(first_confirmation if first_confirmation is not None and
                       first_confirmation<=deadline+1e-9 else None)
    return dict(available=True,sample_period_s=float(delta[0]),threshold_N=float(threshold_N),
        dwell_s=float(dwell_s),release_s=float(release_s),end_s=float(t[-1]),
        first_clear_start_s=first_start,first_clear_confirmation_s=first_confirmation,
        first_clear_confirmation_latency_s=None if first_confirmation is None else first_confirmation-release_s,
        first_clear_alias_role='first_any_within_complete_record; use by_deadline for scheduled outcome',
        first_any_confirmation_s=first_confirmation,
        first_by_deadline_confirmation_s=first_by_deadline,
        first_by_deadline_confirmation_latency_s=None if first_by_deadline is None else first_by_deadline-release_s,
        deadline_s=deadline,not_observed_by_deadline=first_by_deadline is None,
        recontact_episodes_after_first_confirmation=episodes,recontact_count=len(episodes),
        last_observed_contact_exit_s=last_exit,terminal_clear_start_s=final_start,
        terminal_clear_confirmation_s=final_confirm,
        terminal_clear_confirmation_latency_s=None if final_confirm is None else final_confirm-release_s,
        terminal_clear_censored=final_confirm is None,
        clock_role='observed force samples on own grid; <=threshold is a low-force condition, not geometric gap or permanent separation; no interpolation/change to V3')


def evaluate(trace_path, physical_path, protocol, *, original_v3_release=None):
    """Offline evaluator. Raw force uses exact saved substep start+duration=end.

    physical_path is independently recorded robot/force telemetry, used only
    for clock/query cross-check. It is never substituted to improve an outcome.
    """
    th=protocol['thresholds']
    with np.load(trace_path,allow_pickle=False) as z:
        import json
        metadata=json.loads(str(z['metadata']))
        actual=metadata.get('mechanical_release_at')
        if actual is None:actual=metadata.get('release_at')
        if actual is None:
            return dict(available=False,censored=True,reason='no_actual_mechanical_release')
        t=z['time'];frame_contact=z['all_hand_object']
        st=z['substep_time'];duration=z['substep_duration'];rf=z['substep_all_hand_object']
        rt=st+duration
        expected_dt=float(protocol['schedule']['sensor_period_s'])
        final=float(protocol['schedule']['duration_s'])
        deadline=min(final,float(actual)+th['release_deadline_s'])
        coarse_coverage=bool(len(t)==round(final/expected_dt)+1 and
            np.allclose(t,np.arange(len(t))*expected_dt,atol=1e-8,rtol=0)
            and np.isfinite(frame_contact).all())
        raw_coverage=bool(len(st)>0 and st.shape==duration.shape==rf.shape and
            np.isfinite(np.r_[st,duration,rf]).all() and (duration>0).all() and
            abs(st[0])<1e-8 and abs(rt[-1]-final)<1e-8 and
            np.allclose(st[1:],rt[:-1],atol=1e-8,rtol=0) and
            np.max(abs(duration-duration[0]))<1e-8)
        coarse=(clear_series(t,frame_contact,float(actual),th['all_hand_clearance_N'],th['release_clearance_dwell_s'],deadline)
            if coarse_coverage else dict(available=False,censored=True,reason='incomplete_or_nonfinite200Hz_frame_coverage'))
        fine=(clear_series(rt,rf,float(actual),th['all_hand_clearance_N'],th['release_clearance_dwell_s'],deadline)
            if raw_coverage else dict(available=False,censored=True,reason='incomplete_or_nonfinite_raw_substep_coverage'))
    with np.load(physical_path,allow_pickle=False) as z:
        recorded_t=z['physical_time_s'];recorded_f=z['all_hand_target_pressure_N']
        query_check=dict(physical_clock_matches_raw_end=bool(len(recorded_t)==len(rt)+1 and
            np.allclose(recorded_t[1:],rt,atol=1e-8,rtol=0)),
            raw_force_matches_robot_stream_query=bool(len(recorded_f)==len(rf)+1 and
                np.array_equal(recorded_f[1:],rf)))
        # The original evaluator also requires physical exposure. Use its
        # actually returned deadline-limited result; never substitute a late
        # first_any or an unexposed low-force trace into its confirmation window.
        original_latency=(None if original_v3_release is None else
                          original_v3_release.get('physical_clearance_latency_s'))
        original_clock=(None if original_v3_release is None else
                        original_v3_release.get('actual_command_at'))
        confirmation=(None if original_latency is None or original_clock is None else
                      float(original_clock)+float(original_latency))
        if confirmation is not None and confirmation>deadline+1e-9:
            raise ValueError('Supplied original V3confirmation exceeds its registered deadline')
        raw_window=None
        if confirmation is not None and raw_coverage:
            lo=confirmation-th['release_clearance_dwell_s'];hi=confirmation
            threshold=th['all_hand_clearance_N']
            points=(rt>=lo-1e-9)&(rt<=hi+1e-9)
            intervals=(rt>lo+1e-9)&(rt<=hi+1e-9)
            bad=(rf>threshold)&points
            ids=np.flatnonzero(bad);episodes=[]
            if len(ids):
                starts=np.r_[ids[0],ids[1:][np.diff(ids)>1]]
                ends=np.r_[ids[:-1][np.diff(ids)>1],ids[-1]]
                episodes=[dict(first_physical_end_s=float(rt[a]),last_physical_end_s=float(rt[b]),
                    samples=int(b-a+1),touches_window_start=bool(abs(rt[a]-lo)<1e-9),
                    touches_window_end=bool(abs(rt[b]-hi)<1e-9)) for a,b in zip(starts,ends)]
            raw_window=dict(window_s=[float(lo),float(hi)],threshold_N=threshold,
                physical_end_samples=int(points.sum()),peak_target_contact_N=float(rf[points].max()),
                above_threshold_physical_end_samples=int(bad.sum()),
                above_threshold_duration_s=float(np.sum((rf>threshold)[intervals])*np.diff(rt).mean()),
                interval_convention='(confirmation-dwell,confirmation] physical-end rectangles; peak includes both endpoints',
                above_threshold_episodes=episodes,
                interpretation='Original200Hz dwell window not supported by saved integration-segment force observations when violations exist; frame mj_forward and raw post-step cached forces are different queries, not merely decimation')
    return dict(schema='descriptive_first_final_clearance_v1',
        evaluation_only_target_contact=True, online_primitive_reads_contact_identity=False,
        frozen_V3_unchanged=True,observed200Hz=coarse,physical_end2000Hz=fine,
        original_V3_release_metrics_supplied=original_v3_release is not None,
        original_V3_first_confirmation_s=confirmation,
        full_frame_coverage=coarse_coverage,full_raw_substep_coverage=raw_coverage,
        raw_clock=dict(stored_label='mj_step interval start: substep_time',duration_label='substep_duration',
            analysis_label='physical end=substep_time+substep_duration',
            force_query='cached contact-force observations read after mj_step; not a mathematical continuous-contact certificate',
            first_start_s=float(st[0]) if len(st) else None,last_end_s=float(rt[-1]) if len(rt) else None,
            first_clear_confirmation_start_label_s=(fine.get('first_clear_confirmation_s')-float(duration[0]))
                if fine.get('first_clear_confirmation_s') is not None else None),
        physical_stream_crosscheck=query_check,
        raw_inside_original200Hz_first_confirmation_window=raw_window,
        interpretation='First_any, deadline-limited clear, and terminal low-force dwell are separate. <=0.2N low-force is neither geometric gap nor permanent separation. Raw2kHz cached post-step force and200Hz frame force recomputed through additional mj_forward have distinct query semantics; this is neither simple subsampling nor a continuous-physics guarantee. Auxiliary environment contact is not target contact.')
