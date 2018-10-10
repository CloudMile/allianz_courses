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

def do_onehot(catg_ftrs, catg_part, status, data, is_train):
    # mapper = status['mapper']
    # tmp = []
    # for catg_col in catg_ftrs:
    #     if is_train:
    #         result = mapper[catg_col].fit_transform(data[catg_col])
    #     else:
    #         result = mapper[catg_col].transform(data[catg_col])
    #
    #     columns = [f'{catg_col}_{col}' for col in mapper[catg_col].classes_]
    #     if result.shape[1] == 1:
    #         columns = columns[:1]
    #     tmp.append(pd.DataFrame(data=result, columns=columns))
    # tmp = pd.concat(tmp, 1)
    # for col in tmp:
    #     catg_part[col] = tmp[col]
    pass


def do_embedding(catg_ftrs, catg_part, status, data, is_train):
    mapper = status['mapper']
    for emb_col in catg_ftrs:
        if is_train:
            onehot = mapper[emb_col].fit_transform(data[emb_col])
        else:
            onehot = mapper[emb_col].transform(data[emb_col])

        if onehot.shape[1] > 1:
            catg_part[emb_col] = onehot.argmax(1)
        else:
            catg_part[emb_col] = onehot.ravel()
    pass

def do_woe_encoding(catg_ftrs, catg_part, status, data, is_train):

    # def woe_encode(x, label, data):
    #     """Calculate the Weight of Evidence of given categorical feature and label
    #
    #     :param x: Given feature name
    #     :param label: Label name
    #     :param data:
    #     :return: WOE encoded dictionary
    #     """
    #     total_vc = data[label].value_counts().sort_index()
    #
    #     def woe(pipe, total_vc):
    #         # Count by label in this group
    #         group_vc = pipe[label].value_counts().sort_index()
    #
    #         # Some class in the feature is missing, fill zero to missing class
    #         if len(group_vc) < len(total_vc):
    #             for key in total_vc.index:
    #                 if key not in group_vc:
    #                     group_vc[key] = 0.
    #             group_vc = group_vc.sort_index()
    #
    #         # WOE formula
    #         r = ((group_vc + 0.5) / total_vc).values
    #
    #         # Odd ratio => 1 to 0, you can define meaning of each class
    #         return np.log(r[1] / r[0])
    #
    #     return data.groupby(x).apply(lambda pipe: woe(pipe, total_vc))
    #
    # for catg_col in catg_ftrs:
    #     if is_train:
    #         kv = woe_encode(catg_col, 'distress_catg', data)
    #         status['woe_mapper'][catg_col] = kv.to_dict()
    #     else:
    #         kv = pd.Series(status['woe_mapper'][catg_col])
    #     catg_part[f'woe_{catg_col}'] = kv.reindex(data[catg_col]).values
    pass



def do_target_encoding(catg_ftrs, catg_part, status, data, is_train):
    # for catg_col in catg_ftrs:
    #     if is_train:
    #         freq_proportion = data[catg_col].value_counts() / len(data)
    #         catg_part[f'freq_{catg_col}'] = freq_proportion.reindex(data[catg_col]).values
    #         target_mean = data.groupby(catg_col).distress_catg.mean()
    #         catg_part[f'mean_{catg_col}'] = target_mean.reindex(data[catg_col]).values
    #
    #         status['freq_mapper'][catg_col] = freq_proportion.to_dict()
    #         status['mean_mapper'][catg_col] = target_mean.to_dict()
    #     else:
    #         catg_part[f'freq_{catg_col}'] = pd.Series(status['freq_mapper'][catg_col]).reindex(data[catg_col]).values
    #         catg_part[f'mean_{catg_col}'] = pd.Series(status['mean_mapper'][catg_col]).reindex(data[catg_col]).values
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
    # for num_col in num_features:
    #     data[f'{num_col}_degree_2'] = data[num_col] ** 2
    #     data[f'{num_col}_degree_3'] = data[num_col] ** 3

    # for i, col_x in enumerate(num_features):
    #     for j, col_y in enumerate(num_features[i + 1:]):
    #         data[f'cross_{col_x}_x_{col_y}'] = data[col_x] * data[col_y]
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