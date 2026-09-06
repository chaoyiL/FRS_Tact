import json
from execution_trace import ExecutionTrace


def test_trace_keeps_deadline_rejections_and_plan_switch_evidence(tmp_path):
    trace = ExecutionTrace(tmp_path, {'checkpoint':'press/latest.ckpt'})
    path=trace.path
    switch={'new':{'latent':[[1.,2.]],'bases':{'right':[[1.]]}},'old':None}
    trace.write({'obs_seq':5,'deadline':100.215,'runtime':{'plan_switch':switch},
                 'receipt':{'status':'rejected','scheduled_count':0,'target_timestamp':None}})
    trace.close()
    lines=[json.loads(x) for x in path.read_text().splitlines()]
    assert lines[0]['type']=='session'
    assert lines[1]['obs_seq']==5
    assert lines[1]['receipt']['scheduled_count']==0
    assert lines[1]['runtime']['plan_switch']==switch
    again=ExecutionTrace(tmp_path,{})
    try: assert again.path != path
    finally: again.close()
