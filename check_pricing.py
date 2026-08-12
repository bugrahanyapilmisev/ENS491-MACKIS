
import urllib.request, json
req = urllib.request.Request('https://openrouter.ai/api/v1/models')
data = json.loads(urllib.request.urlopen(req).read().decode())['data']
for m in data:
    if m['id'] in ['qwen/qwen3-235b-a22b', 'qwen/qwen3-30b-a3b']:
        p = m['pricing']
        print(m['id'], 'Prompt:', float(p['prompt'])*1000000, 'Completion:', float(p['completion'])*1000000)

