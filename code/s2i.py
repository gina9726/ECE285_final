import json
import pdb

phase = 'test'
model = 'baseline_lr6e-4'
ep = 23000
res_json = 'checkpoint/%s/ep-%d_%s.json' % (model, ep, phase)

print('loading json file...')
if phase == 'test':
    phase = 'val'

with open(res_json) as data_file:
    data = json.load(data_file)

for i in xrange(len(data)):
    print i
    data[i]['question_id'] = int(data[i]['question_id'])

output_json = 'v2_OpenEnded_mscoco_%s2014_real_results.json' % (phase)
dd = json.dump(data,open(output_json,'w'))
