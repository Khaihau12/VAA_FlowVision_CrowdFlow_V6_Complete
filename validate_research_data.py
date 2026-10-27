"""Small reproducibility check that does not require Streamlit."""
from src.data import load_all
from src.zones import fit_pca_zones, occupancy_table

D = load_all()
model = fit_pca_zones(D["gt"])
gt_test = D["gt"][D["gt"]["split"] == "test"]
pred_test = D["fusion"][D["fusion"]["split"] == "test"]

gt_mean = occupancy_table(gt_test, model).mean().round(4).to_dict()
pred_mean = occupancy_table(pred_test, model).mean().round(4).to_dict()

expected_gt = {"Zone A": 7.575, "Zone B": 10.775, "Zone C": 1.75}
expected_pred = {"Zone A": 5.1125, "Zone B": 8.55, "Zone C": 1.4125}

print("GT mean occupancy:", gt_mean)
print("Pred mean occupancy:", pred_mean)
assert gt_mean == expected_gt, (gt_mean, expected_gt)
assert pred_mean == expected_pred, (pred_mean, expected_pred)

p12 = D["scorecard"].query("pair == 'Pair12'").iloc[0]
assert round(float(p12["Tracking_IDF1"]), 4) == 0.4684
assert round(float(p12["EntryExit_F1"]), 4) == 0.6875
assert round(float(p12["Flow_Event_F1"]), 4) == 0.6875
print("Pair12 key metrics: OK")
print("REPRODUCIBILITY CHECK: PASS")
