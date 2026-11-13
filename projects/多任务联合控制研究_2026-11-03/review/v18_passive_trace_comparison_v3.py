"""Complete passive bridge, with an explicit measured-wall-time whitelist.

The initial all-column failure is retained. No physics, public measurement,
controller numeric value, status, or full debug packet is discarded.
"""
from pathlib import Path
import hashlib,json
import numpy as np

TIMING_COLUMNS=('control_ms','actor_callback_ms','full_control_loop_ms','observation_ms','physics_advance_ms')
DEBUG_TIMING_KEYS=('qp','native_inverse','whole_decision')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def debug_payload_without_declared_timing(text):
    packet=json.loads(str(text))
    if not isinstance(packet,dict):raise ValueError('debug object required')
    timing=packet.get('solver_timing_ms')
    if timing is not None:
        if not isinstance(timing,dict):raise ValueError('timing dictionary required')
        for key in DEBUG_TIMING_KEYS:
            if key in timing:
                value=timing[key]
                if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<0:
                    raise ValueError('invalid measured wall time')
                timing[key]='__validated_measured_wall_time__'
    return json.dumps(packet,ensure_ascii=False,allow_nan=True,sort_keys=True,separators=(',',':')).encode('utf-8')


def compare_arrays(a,b):
    checks={};keys=set(a)|set(b)
    same_keys=set(a)==set(b)
    for name in sorted(keys):
        if name not in a or name not in b:
            checks[name]=dict(passed=False,reason='missing_column');continue
        x,y=a[name],b[name]
        if name=='metadata':continue
        if name in TIMING_COLUMNS:
            valid=x.shape==y.shape and x.dtype==y.dtype and x.dtype.kind in 'fi'
            valid=bool(valid and np.isfinite(x).all() and np.isfinite(y).all() and (x>=0).all() and (y>=0).all())
            checks[name]=dict(passed=valid,rule='same_shape_dtype_finite_nonnegative_measured_wall_time_only');continue
        if name=='debug_json':
            valid=x.shape==y.shape and x.dtype.kind=='U' and y.dtype.kind=='U'
            try:
                valid=bool(valid and all(debug_payload_without_declared_timing(u)==debug_payload_without_declared_timing(v) for u,v in zip(x.flat,y.flat)))
            except (ValueError,TypeError,json.JSONDecodeError):valid=False
            checks[name]=dict(passed=valid,rule='all_debug_fields_exact_except_three_explicit_solver_wall_time_leaves');continue
        checks[name]=dict(passed=bool(x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes()),rule='exact_shape_dtype_bytes')
    return dict(passed=same_keys and bool(checks) and all(c['passed'] for c in checks.values()),same_column_keys=same_keys,checks=checks)


def compare(actual,original):
    actual=Path(actual);original=Path(original)
    with np.load(actual/'trace.npz',allow_pickle=False) as a,np.load(original/'trace.npz',allow_pickle=False) as b:
        result=compare_arrays(a,b)
    return dict(schema='v18_complete_passive_bridge_walltime_only_v3',**result,
        actual=str(actual),original=str(original),actual_trace_sha256=sha(actual/'trace.npz'),original_trace_sha256=sha(original/'trace.npz'),
        excluded_provenance_column='metadata',measured_timing_columns=list(TIMING_COLUMNS),
        debug_timing_paths=['solver_timing_ms.'+x for x in DEBUG_TIMING_KEYS],
        no_controller_or_measurement_fields_removed=True,complete_16s_comparison=True)
