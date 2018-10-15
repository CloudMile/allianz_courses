import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

from keras import metrics

def do_binning(catg_ftrs, status, data, is_train):
    def quartile_binning(x):
        bins = np.percentile(x, range(0, 100, 25))[1:].tolist()
        iqr_x_150 = (bins[-1] - bins[0]) * 1.5
        bins = [bins[0] - iqr_x_150] + bins + [bins[-1] + iqr_x_150]
        result = pd.Series(np.digitize(x, bins)).map(pd.Series([0, 1, 2, 3, 4, 0])).values
        return result, bins

    for col in ('tenure', 'MonthlyCharges', 'TotalCharges'):
        binned_name = f'binn_{col}'
        if is_train:
            result, bins = quartile_binning(data[col])
            status['binn_mapper'][binned_name] = bins
            data[binned_name] = result
        else:
            bins = status['binn_mapper'][binned_name]
            data[binned_name] = pd.Series(np.digitize(data[col], bins))\
                                  .map(pd.Series([0, 1, 2, 3, 4, 0])).values

        catg_ftrs.append(binned_name)
    pass

def do_onehot(catg_ftrs, encoded_ftrs, status, data, is_train):
    mapper = status['mapper']
    tmp = []
    for catg_col in catg_ftrs:
        if is_train:
            result = mapper[catg_col].fit_transform(data[catg_col])
        else:
            result = mapper[catg_col].transform(data[catg_col])

        columns = [f'{catg_col}_{col}' for col in mapper[catg_col].classes_]
        if result.shape[1] == 1:
            columns = columns[:1]
        tmp.append(pd.DataFrame(data=result, columns=columns))
    tmp = pd.concat(tmp, 1)
    for col in tmp:
        encoded_ftrs[col] = tmp[col]
    pass


def do_embedding(catg_ftrs, encoded_ftrs, status, data, is_train):
    # mapper = status['mapper']
    # for emb_col in catg_ftrs:
    #     if is_train:
    #         onehot = mapper[emb_col].fit_transform(data[emb_col])
    #     else:
    #         onehot = mapper[emb_col].transform(data[emb_col])
    #
    #     if onehot.shape[1] > 1:
    #         encoded_ftrs[emb_col] = onehot.argmax(1)
    #     else:
    #         encoded_ftrs[emb_col] = onehot.ravel()
    # return encoded_ftrs
    pass

def do_woe_encoding(catg_ftrs, encoded_ftrs, status, data, is_train):

    def woe_encode(x, label, data):
        """Calculate the Weight of Evidence of given categorical feature and label

        :param x: Given feature name
        :param label: Label name
        :param data:
        :return: WOE encoded dictionary
        """
        total_vc = data[label].value_counts().sort_index()

        def woe(pipe, total_vc):
            # Count by label in this group
            group_vc = pipe[label].value_counts().sort_index()

            # Some class in the feature is missing, fill zero to missing class
            if len(group_vc) < len(total_vc):
                for key in total_vc.index:
                    if key not in group_vc:
                        group_vc[key] = 0.
                group_vc = group_vc.sort_index()

            # WOE formula
            r = ((group_vc + 0.5) / total_vc).values

            # Odd ratio => 1 to 0, you can define meaning of each class
            return np.log(r[1] / r[0])

        return data.groupby(x).apply(lambda pipe: woe(pipe, total_vc))

    for catg_col in catg_ftrs:
        if is_train:
            kv = woe_encode(catg_col, 'Exited', data)
            status['woe_mapper'][catg_col] = kv.to_dict()
        else:
            kv = pd.Series(status['woe_mapper'][catg_col])
        encoded_ftrs[f'woe_{catg_col}'] = kv.reindex(data[catg_col]).values
    pass

def do_entropy_encoding(catg_ftrs, encoded_ftrs, status, data, is_train):
    # def entropy_encode(x, label, data):
    #     """Calculate the entropy of given categorical feature and label
    #
    #     :param x: Given feature name
    #     :param label: Label name
    #     :param data:
    #     :return: WOE encoded dictionary
    #     """
    #     def entropy(pipe):
    #         # Count by label in this group
    #         group_vc = pipe[label].value_counts()
    #         # Only one class in label, the entropy equal to zero
    #         if len(group_vc) <= 1:
    #             return 0
    #
    #         return -(group_vc / len(pipe)).map(lambda e: e * np.log(e)).sum()
    #
    #     class_proba = data.groupby(x).size() / len(data)
    #     class_entropy = data.groupby(x).apply(entropy)
    #     return class_proba.reindex(class_entropy.index).values * class_entropy
    #
    # for catg_col in catg_ftrs:
    #     if is_train:
    #         kv = entropy_encode(catg_col, 'distress_catg', data)
    #         status['entropy_mapper'][catg_col] = kv.to_dict()
    #     else:
    #         kv = pd.Series(status['entropy_mapper'][catg_col])
    #     encoded_ftrs[f'entropy_{catg_col}'] = kv.reindex(data[catg_col]).values
    pass

def do_target_encoding(catg_ftrs, encoded_ftrs, status, data, is_train):
    for catg_col in catg_ftrs:
        if is_train:
            freq_proportion = data[catg_col].value_counts() / len(data)
            encoded_ftrs[f'freq_{catg_col}'] = freq_proportion.reindex(data[catg_col]).values
            target_mean = data.groupby(catg_col).Exited.mean()
            encoded_ftrs[f'mean_{catg_col}'] = target_mean.reindex(data[catg_col]).values

            status['freq_mapper'][catg_col] = freq_proportion.to_dict()
            status['mean_mapper'][catg_col] = target_mean.to_dict()
        else:
            encoded_ftrs[f'freq_{catg_col}'] = pd.Series(status['freq_mapper'][catg_col]).reindex(data[catg_col]).values
            encoded_ftrs[f'mean_{catg_col}'] = pd.Series(status['mean_mapper'][catg_col]).reindex(data[catg_col]).values
    pass


def do_norm(num_features, status, data, is_train):
    num_part = data[num_features].copy()
    if is_train:
        scaler = StandardScaler()
        status['scaler'] = scaler
        num_part = pd.DataFrame(data=scaler.fit_transform(num_part), columns=num_part.columns)
    else:
        scaler = status['scaler']
        num_part = pd.DataFrame(data=scaler.transform(num_part), columns=num_part.columns)
    return num_part


def do_nth_order_polynominal(num_features, data):
    for num_col in num_features:
        data[f'{num_col}_degree_2'] = data[num_col] ** 2
        data[f'{num_col}_degree_3'] = data[num_col] ** 3
    pass
