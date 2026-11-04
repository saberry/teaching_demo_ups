import matplotlib.pyplot as plt
import pandas as pd
import shap
from xgboost import XGBRegressor

model = XGBRegressor()
model.load_model("xgb_price_change.json")

options = pd.read_csv("ticket_features.csv")
options = options[options["target"] != 0].reset_index(drop=True)
X = options[model.get_booster().feature_names]

explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

shap.plots.bar(shap_values, show=False)
plt.title("Average impact on predicted log price change")
plt.tight_layout()
plt.savefig("shap_importance.png", dpi=150)
plt.close()

shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=150)
plt.close()

top_feature = abs(shap_values.values).mean(axis=0).argmax()
shap.plots.scatter(shap_values[:, top_feature], color=shap_values, show=False)
plt.tight_layout()
plt.savefig("shap_dependence_top_feature.png", dpi=150)
plt.close()

shap.plots.waterfall(shap_values[0], show=False)
plt.tight_layout()
plt.savefig("shap_single_prediction.png", dpi=150)
plt.close()
