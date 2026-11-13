"""One precommit reference selector; the real lifecycle/actor stays its owner."""
from copy import deepcopy
import numpy as np
from v13_a1.codec import Reference21,finite_array,REFERENCE_AXES21
from v14_coord.design import reference_rates
from v25_response_identification.runtime import ProbeCoordinator as ReceiptCoordinator,plain
from .design import ReferenceProbeSpec,CONTROL_PERIOD_S

class FullReferenceCoordinator(ReceiptCoordinator):
    def __init__(self,nominal,spec):
        if type(spec) is not ReferenceProbeSpec:raise TypeError('Typed V26 reference program required')
        self.nominal_supervisor=nominal;self.spec=spec;self.records=[]
        self.carrier21=None;self.target21=None;self.origin_generation=None;self.terminal_status=None
        self.component_support=();self.origin_exposure_status=None

    def action(self,**kwargs):
        proposed,debug=self.nominal_supervisor.action(**kwargs);now=float(kwargs['frame'].time)
        if now<self.spec.origin_s-1e-9:return proposed,debug
        record=self.nominal_supervisor.decision_evidence[-1]
        lo=finite_array(record['actual_reference_lower21'],(21,),'actualTASKlower')
        hi=finite_array(record['actual_reference_upper21'],(21,),'actualTASKupper')
        if np.any(lo>hi):raise ValueError('Empty actual reference authority')
        if self.carrier21 is None:
            if abs(now-self.spec.origin_s)>1e-9:raise ValueError('Program must start at registered origin')
            self.carrier21=proposed.values.copy();self.target21=self.carrier21.copy();self.origin_generation=int(kwargs['generation'])
            components=[]
            for axis,sign in self.spec.components:
                margin=max(0.,float(hi[axis]-self.carrier21[axis] if sign>0 else self.carrier21[axis]-lo[axis]))
                amplitude=self.spec.amplitude_fraction*min(reference_rates('release')[axis]*CONTROL_PERIOD_S,margin)
                requested=self.carrier21[axis]+sign*amplitude
                supported=amplitude>0 and requested!=self.carrier21[axis]
                if supported:self.target21[axis]=requested
                components.append(dict(axis=axis,name=REFERENCE_AXES21[axis],sign=sign,
                    origin_direction_margin=margin,raw_amplitude=amplitude,
                    target_increment=requested-self.carrier21[axis] if supported else 0.,
                    supported=supported,reason=None if supported else 'ZERO_DIRECTION_AUTHORITY_OR_REFERENCE_ROUNDING'))
            self.component_support=tuple(components);n=sum(c['supported'] for c in components)
            self.origin_exposure_status=('ZERO_CONTROL' if not components else 'FULLY_EXPOSED' if n==len(components) else 'UNEXPOSED' if n==0 else 'PARTIAL_UNSUPPORTED')
        if self.terminal_status is None and int(kwargs['generation'])!=self.origin_generation:self.terminal_status='CANCELLED_MODE_GENERATION_CHANGED'
        if self.terminal_status is None and now>=self.spec.origin_s+self.spec.hold_s-1e-9:self.terminal_status='PROBE_WINDOW_COMPLETED_'+self.origin_exposure_status
        holding=self.terminal_status is None
        # Crucial: UNEXPOSED holds exactly the zero target. It does not resume
        # the adaptive policy until the common hold window ends or preempts.
        values=np.clip(self.target21,lo,hi) if holding else proposed.values.copy()
        emitted=Reference21(values,now);changed=values.tobytes()!=proposed.values.tobytes()
        certified=record.get('geometry_proposal_accepted') is True and (record.get('emitted_hierarchy_score') or {}).get('available') is True
        active=holding and self.origin_exposure_status!='UNEXPOSED'
        row=dict(time_s=now,arm=self.spec.arm,role=self.spec.role,origin_s=self.spec.origin_s,
            active=active,carrier_hold_active=holding,
            status=('CARRIER_HOLD_'+self.origin_exposure_status) if holding else self.terminal_status,
            origin_exposure_status=self.origin_exposure_status,component_support=plain(self.component_support),
            requested_components=[dict(axis=a,name=REFERENCE_AXES21[a],sign=s) for a,s in self.spec.components],
            origin_generation=self.origin_generation,current_generation=int(kwargs['generation']),
            carrier_at_origin21=plain(self.carrier21),raw_probe_target21=plain(self.target21),
            raw_probe_increment21=(self.target21-self.carrier21).tolist(),
            nominal_proposal_reference21=proposed.values.tolist(),action_emitted_reference21=values.tolist(),
            actual_TASK_lower21=lo.tolist(),actual_TASK_upper21=hi.tolist(),
            requested_target_clipped_by_fresh_TASK=holding and not np.array_equal(values,self.target21),
            nominal_reference_unchanged=not changed,nominal_task_certificate_available=bool(certified),
            nominal_task_certificate_applies_to_emitted_reference=bool(certified and not changed),
            full_declared_program_exposed=self.origin_exposure_status in ('ZERO_CONTROL','FULLY_EXPOSED'),
            probe_future_response_prediction_available=False,actual_receipt=None)
        self.records.append(row);result=deepcopy(debug)
        result['response_probe']={k:v for k,v in row.items() if k not in ('actual_TASK_lower21','actual_TASK_upper21')}
        return emitted,result
