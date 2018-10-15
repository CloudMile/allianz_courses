import numpy as np
import pandas as pd
from keras.layers import Concatenate
from keras.layers import Dense, Dropout, Embedding, Flatten, Input
from keras.models import Model
from sklearn.preprocessing import StandardScaler


def do_binning(catg_ftrs, status, data, is_train):
    # def quartile_binning(x):
    #     bins = np.percentile(x, range(0, 100, 25))[1:].tolist()
    #     iqr_x_150 = (bins[-1] - bins[0]) * 1.5
    #     bins = [bins[0] - iqr_x_150] + bins + [bins[-1] + iqr_x_150]
    #     result = pd.Series(np.digitize(x, bins)).map(pd.Series([0, 1, 2, 3, 4, 0])).values
    #     return result, bins
    #
    # for col in [f'x{i}' for i in range(1, 84) if i != 80]:
    #     binned_name = f'binn_{col}'
    #     if is_train:
    #         result, bins = quartile_binning(data[col])
    #         status['binn_mapper'][binned_name] = bins
    #         data[binned_name] = result
    #     else:
    #         bins = status['binn_mapper'][binned_name]
    #         data[binned_name] = pd.Series(np.digitize(data[col], bins))\
    #                               .map(pd.Series([0, 1, 2, 3, 4, 0])).values
    #
    #     catg_ftrs.append(binned_name)
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


def do_rfm(catg_ftrs, encoded_ftrs, status, data, is_train):
    if is_train:
        def rfm(pipe):
            # ret = {}
            pipe['rfm_all_freq'] = np.arange(len(pipe)) + 1
            pipe['rfm_all_mean'] = pipe.distress_num.cumsum() / (np.arange(len(pipe)) + 1)
            return pipe[['rfm_all_freq',
                         'rfm_all_mean'
                         ]]

        added_features = data.groupby('Company').apply(rfm)
        added_features.insert(0, 'Company', data.Company.values)
        status['rfm_mapper'] = added_features.groupby('Company', as_index=False).last().to_dict('records')
    else:
        rfm_mapper = status['rfm_mapper']
        added_features = pd.DataFrame(data=rfm_mapper).set_index('Company').reindex(data.Company)

    encoded_ftrs['rfm_all_freq'] = added_features['rfm_all_freq'].values
    encoded_ftrs['rfm_all_mean'] = added_features['rfm_all_mean'].values
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
            kv = woe_encode(catg_col, 'distress_catg', data)
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
            target_mean = data.groupby(catg_col).distress_catg.mean()
            encoded_ftrs[f'mean_{catg_col}'] = target_mean.reindex(data[catg_col]).values

            status['freq_mapper'][catg_col] = freq_proportion.to_dict()
            status['mean_mapper'][catg_col] = target_mean.to_dict()
        else:
            encoded_ftrs[f'freq_{catg_col}'] = pd.Series(status['freq_mapper'][catg_col]).reindex(data[catg_col]).values
            encoded_ftrs[f'mean_{catg_col}'] = pd.Series(status['mean_mapper'][catg_col]).reindex(data[catg_col]).values
    pass


def do_norm(num_features, encoded_ftrs, status, data, is_train):
    num_part = data[num_features].copy()
    if is_train:
        scaler = StandardScaler()
        status['scaler'] = scaler
        num_part = pd.DataFrame(data=scaler.fit_transform(num_part), columns=num_part.columns)
    else:
        scaler = status['scaler']
        num_part = pd.DataFrame(data=scaler.transform(num_part), columns=num_part.columns)

    for col in num_part:
        encoded_ftrs[col] = num_part[col].values

def do_nth_order_polynominal(num_features, data):
    # for num_col in num_features:
    #     data[f'{num_col}_degree_2'] = data[num_col] ** 2
    #     data[f'{num_col}_degree_3'] = data[num_col] ** 3
    pass


def get_model(input_dim, catg_ftrs):
    emb_unique_len = {
        'Company': 422,
        'x80': 37
    }
    inputs = Input(shape=(input_dim,))
    emb_inputs = [Input(shape=(1,)) for col in catg_ftrs]
    all_inputs = [inputs] + emb_inputs

    embeddings = [Flatten(name=name)(Embedding(emb_unique_len[name], 8, input_length=1)(emb))
                  for name, emb in zip(catg_ftrs, emb_inputs)]

    concat = Concatenate(axis=1)([inputs] + embeddings)
    nets = Dense(units=64, activation='selu')(concat)
    nets = Dropout(.3)(nets)
    nets = Dense(units=32, activation='selu')(nets)
    nets = Dropout(.3)(nets)
    nets = Dense(units=16, activation='selu')(nets)
    nets = Dropout(.3)(nets)
    nets = Dense(units=1, activation='linear')(nets)

    model = Model(inputs=all_inputs, outputs=nets)
    model.summary()
    model.compile(optimizer='adam', loss='mse')
    return model

# # Split train valid
# def split_fn(pipe):
#     lens = len(pipe)
#     ret = np.ones(lens)
#     if lens >= 5:
#         split_size = int(lens * 0.36)
#         ret[-split_size:] = 0
#     return ret

# raw['distress_catg'] = (raw.distress_num <= -0.5).astype(int)
# raw['is_train'] = np.concatenate(raw.groupby('Company').apply(split_fn).values)
# raw_vl = raw.query("is_train == 0").drop('is_train', 1)
# raw = raw.query("is_train == 1").drop('is_train', 1)
# raw.head()

# def get_embedding_model(input_dim):
#     emb_unique_len = {
#         'x80': 37
#     }
#     inputs = Input(shape=(input_dim,))
#     emb_inputs = [Input(shape=(1,)) for col in emb_unique_len]
#     all_inputs = [inputs] + emb_inputs
#
#     embeddings = [Flatten(name=name)(Embedding(emb_unique_len[name], 8, input_length=1)(emb))
#                   for name, emb in zip(embedding_features, emb_inputs)]
#
#     concat = Concatenate(axis=1)([inputs] + embeddings)
#     nets = Dense(units=32, activation='relu')(concat)
#     nets = Dense(units=16, activation='relu')(nets)
#     nets = Dense(units=16, activation='relu')(nets)
#     nets = Dense(units=1, activation='sigmoid', )(nets)
#     model = Model(inputs=all_inputs, outputs=nets)
#     model.summary()
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model


# K.clear_session()
#
# input_dim = len(set(tr_x.columns) - set(embedding_features))
# model = get_embedding_model(input_dim)
#
#
# def make_x(input_x):
#     return [input_x.drop(embedding_features, 1)] + [input_x[col][:, None] for col in embedding_features]
#
#
# hist = model.fit(make_x(tr_x), tr_y,
#                  validation_data=(make_x(vl_x), vl_y),
#                  batch_size=100, epochs=30)
# plot_result(hist)
# draw_roc_curve(vl_y, model.predict(make_x(vl_x)))
# 依照所有的閥值(切割100等分)算出F score, 找出分數最高的閥值

