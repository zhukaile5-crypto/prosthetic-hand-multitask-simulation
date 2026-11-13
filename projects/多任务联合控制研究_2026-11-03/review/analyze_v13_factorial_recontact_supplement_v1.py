"""Read-only suffix-clearance and saved full-reset-state supplement."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'review/v13_release_factorial_recontact_supplement_v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    if OUT.exists():raise FileExistsError('Keep previous supplement immutable')
    folders={'both':ROOT/'output/v13_release_factorial_bridge_v1/both',**{k:ROOT/'output/v13_release_factorial_other3_v1'/k for k in ('fingers_only','lift_only','neither')}}
    inputs={};resets={};support_times={};detail=None
    for name,folder in folders.items():
        state=folder/'reset_integration_state.npz';path=folder/'trace.npz';inputs[str(state)]=sha(state);inputs[str(path)]=sha(path)
        with np.load(state,allow_pickle=False) as z:resets[name]=(z['state'].copy(),int(z['state_spec']))
        with np.load(path,allow_pickle=False) as z:st=z['substep_time'].copy();du=z['substep_duration'].copy();support=z['substep_support'].copy();f=z['substep_all_hand_object'].copy();t=z['time'].copy();ff=z['all_hand_object'].copy()
        end=st+du;release=st>=12.-1e-9;indices=np.flatnonzero(release);i=int(indices[np.argmax(support[release])]);support_times[name]=dict(peak_N=float(support[i]),peak_END_s=float(end[i]))
        if name=='both':
            declared=json.loads((folder/'diagnostics.json').read_text('utf-8'))['low_force_clearance'];first=float(declared['raw_integration_end_first_by_deadline']);after=np.flatnonzero((end>first+1e-9)&(f>.2));peak=int(after[np.argmax(f[after])]);last=int(np.flatnonzero((end>=12)&(f>.2))[-1]);low_start=float(end[last+1]);confirm=np.flatnonzero((end>=low_start+.2-1e-9)&(end<=15.5+1e-9));frame=np.flatnonzero((t>float(declared['frame200Hz_first_by_deadline'])+1e-9)&(ff>.2))
            detail=dict(original_first_rawEND_confirmation_s=first,original_first_frame_confirmation_s=declared['frame200Hz_first_by_deadline'],
                first_recontact_END_s=float(end[after[0]]),last_recontact_END_s=float(end[after[-1]]),total_postconfirmation_exposure_above_clearance_s=float(du[after].sum()),
                recontact_peak_N=float(f[peak]),recontact_peak_END_s=float(end[peak]),frame_recontact_count=len(frame),frame_recontact_peak_N=float(ff[frame].max()),
                final_observed_low_force_suffix_first_END_s=low_start,final_suffix_200ms_confirmation_END_s=None if not len(confirm) else float(end[confirm[0]]),observed_through_s=float(end[-1]))
    first,spec=resets['both'];reset_equal={name:sp==spec and v.dtype==first.dtype and v.shape==first.shape and v.tobytes()==first.tobytes() for name,(v,sp) in resets.items()}
    result=dict(schema='v13_release_factorial_recontact_supplement_v1',input_sha256=inputs,checker_sha256=sha(__file__),
        saved_reset_integration_state_spec=spec,all_four_saved_reset_states_byte_equal=all(reset_equal.values()),reset_comparisons=reset_equal,
        support_peaks=support_times,both=detail,
        scope='Retrospective fixed-input physics diagnosis only. Original first-dwell metric and all archived outcomes unchanged. A final observed low-force suffix through16s is not permanent/geometric separation or a new algorithm success endpoint.',physics_steps=0)
    if not all(sha(p)==h for p,h in inputs.items()):raise ValueError('Input changed during supplement')
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
