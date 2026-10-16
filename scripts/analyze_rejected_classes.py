import csv
from collections import Counter

# Oversized debris rejects
with open('data/ground_truth_audit/oversized_debris/oversized_debris_list.csv') as f:
    ov = list(csv.DictReader(f))
with open('data/ground_truth_audit/oversized_debris_decisions.csv') as f:
    ov_dec = {row['image']: row['decision'] for row in csv.DictReader(f)}

ov_rejected_classes = Counter()
for row in ov:
    vis = f"{row['split']}_{row['image']}.jpg"
    if ov_dec.get(vis) == 'reject':
        ov_rejected_classes[row['class']] += 1
print('Oversized-debris rejects by class:', ov_rejected_classes)

# Tiny box rejects
with open('data/ground_truth_audit/tiny_boxes/tiny_boxes_list.csv') as f:
    tiny = list(csv.DictReader(f))
with open('data/ground_truth_audit/tiny_boxes_decisions.csv') as f:
    tiny_dec = {row['image']: row['decision'] for row in csv.DictReader(f)}

seen = set()
tiny_rejected_classes = Counter()
for row in tiny:
    vis = f"{row['split']}_{row['image']}.jpg"
    if vis in seen:
        continue
    seen.add(vis)
    if tiny_dec.get(vis) == 'reject':
        tiny_rejected_classes[row['class']] += 1
print('Tiny-box rejects by class:', tiny_rejected_classes)