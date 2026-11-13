"""Literal paired-prefix comparison when BOTH runs have the same instrument.

No metadata projection, field omission, numerical tolerance, or simulation.
Unicode array storage widths may differ because of post-release records; each
actual string payload is compared byte-for-byte in UTF8 instead of padding.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from review.run_v16_release_small_v1 import PHYSICAL_COLUMNS
from v17_contact_transfer.motion_domain_sensors import ADDED_FIELDS


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def compare(full,holding):
    full=Path(full);holding=Path(holding);checks={};meta_ok=True;packets=0
    with np.load(full/'trace.npz',allow_pickle=False) as a,np.load(holding/'trace.npz',allow_pickle=False) as b:
        ma=a['time']<12.-1e-9;mb=b['time']<12.-1e-9
        for name in tuple(PHYSICAL_COLUMNS)+('raw_patches_json',):
            if name not in a or name not in b:
                checks[name]=dict(equal=False,reason='missing_required_column');continue
            aa=a[name][ma];bb=b[name][mb];same=aa.shape==bb.shape
            if aa.dtype.kind=='U' and bb.dtype.kind=='U':
                same=bool(same and all(str(x).encode('utf-8')==str(y).encode('utf-8') for x,y in zip(aa.flat,bb.flat)))
                rule='exact_UTF8_payload_including_all_JSON_fields_and_metadata'
            else:
                same=bool(same and aa.dtype==bb.dtype and aa.tobytes()==bb.tobytes())
                rule='exact_dtype_shape_bytes'
            checks[name]=dict(equal=same,rule=rule)
            if name=='legacy_common_sensor_json':
                if aa.shape!=bb.shape:meta_ok=False
                else:
                    for x,y in zip(aa.flat,bb.flat):
                        xx=json.loads(str(x));yy=json.loads(str(y));packets+=1
                        if any(key not in xx or key not in yy or xx[key]!=yy[key] for key in ADDED_FIELDS):
                            meta_ok=False
    return dict(schema='v17b_same_instrument_literal_prefix_v1',
        passed=all(c['equal'] for c in checks.values()) and meta_ok,checks=checks,
        columns=len(checks),additional_domain_metadata_exact=meta_ok,metadata_packets_compared=packets,
        full=str(full),holding=str(holding),scope='strictly_before12',no_fields_removed=True,
        full_trace_sha256=sha(full/'trace.npz'),holding_trace_sha256=sha(holding/'trace.npz'))
