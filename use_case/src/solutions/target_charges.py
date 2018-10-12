g = sns.FacetGrid(data=raw_telco, hue='Churn', height=4, aspect=1.5)
g.map(sns.distplot, 'TotalCharges', kde=False).add_legend()