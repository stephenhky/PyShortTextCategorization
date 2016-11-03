import pandas as pd
from collections import defaultdict

def retrieve_data_as_dict(filepath):
    df = pd.read_csv(filepath)
    category_col, descp_col = df.columns.values.tolist()
    shorttextdict = defaultdict(lambda : [])
    for category, descp in zip(df[category_col], df[descp_col]):
        shorttextdict[category] += [descp]
    return dict(shorttextdict)