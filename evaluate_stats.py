import json
with open('public/tgn_stats.json') as f:
    stats = json.load(f)
print("threshold_clean:", stats['threshold'])
