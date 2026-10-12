"""Hand labels for data/labeling/crops (60 largest merged tracks of data/mall_clip.mp4).

Index = position in data/labeling/labels.csv (contact sheets sheet0/1.png). Labelled by
eye by one annotator on ~90x45 px crops, so treat as a rough check, not ground truth.
1 = carrying a bag (shopping bag, handbag, shoulder bag, backpack), 0 = no visible bag.
Left out: 12, 40, 42, 44, 52 (unsure) and 50 (a Christmas tree, not a person).
"""
BAG = {3, 6, 8, 10, 11, 13, 15, 17, 19, 24, 25, 28, 29, 30, 35, 38, 45, 48, 49, 59}
UNSURE = {12, 40, 42, 44, 52}
NOT_PERSON = {50}
# Same red-haired promoter fragmented into several tracks (visible on the sheets):
SAME_PERSON = {2, 4, 7, 12, 22, 31, 37, 53, 57}
LABELS = {i: int(i in BAG) for i in range(60) if i not in UNSURE | NOT_PERSON}
