import pandas as pd
import os
import dill
import numpy as np
from datetime import datetime

from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
#from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from category_encoders import TargetEncoder


def merge_datasets(df_h, df_s):
    # ga_hits
    event_target = ('sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
                    'sub_custom_question_submit_click', 'sub_call_number_click', 'sub_callback_submit_click',
                    'sub_submit_success', 'sub_car_request_submit_click')
    df0 = df_h[df_h.event_action.isin(event_target)]

    df0.loc[:, 'event_action'] = 1
    df0 = df0.drop_duplicates()

    # ga_sessions
    df1 = df_s.drop_duplicates()

    # unification
    df = df1.merge(df0, on='session_id', how='left')
    df.event_action = df.event_action.fillna(value=0)
    return df.drop('event_action', axis=1), df['event_action']


def filter_data(df):
    columns_for_drop = ['session_id', 'client_id', 'device_model', 'device_os', 'utm_keyword',
                        'device_screen_resolution']
    return df.drop(columns_for_drop, axis=1)


def my_iputer(df):
    df = df.copy()
    fill_list = ['utm_source', 'utm_adcontent', 'device_brand', 'utm_campaign']
    for key in fill_list:
        df[key] = df[key].fillna(value='other')
        df[key] = df[key].fillna(value=df[key].mode()[0])
    return df


def add_time_features(df):
    import pandas as pd

    df = df.copy()
    # form datetime feature
    df['visit_datetime'] = pd.to_datetime(df['visit_date'] + ' ' + df['visit_time'])

    # add new time features
    df['visit_dayofweek'] = df['visit_datetime'].dt.dayofweek
    df['visit_day'] = df['visit_datetime'].dt.day
    df['visit_month'] = df['visit_datetime'].dt.month
    df['visit_hour'] = df['visit_datetime'].dt.hour

    df = df.drop(['visit_date', 'visit_time', 'visit_datetime'], axis=1)
    return df


def add_binary_features(df):
    df = df.copy()
    df['is_Russia'] = df['geo_country'].apply(lambda x: 1 if x == 'Russia' else 0).astype('int8')
    df['is_mobile'] = df['device_category'].apply(lambda x: 1 if x in ['mobile', 'tablet'] else 0).astype('int8')

    df = df.drop(['geo_country', 'device_category'], axis=1)
    return df


def remove_visit_number_outliers(df):
    q95 = df.visit_number.quantile(0.95)
    q05 = df.visit_number.quantile(0.05)
    boundaries = (q05 - 1.5 * (q95 - q05), q95 + 1.5 * (q95 - q05))
    df = df.copy()

    df.loc[df['visit_number'] < boundaries[0], 'visit_number'] = round(boundaries[0])
    df.loc[df['visit_number'] > boundaries[1], 'visit_number'] = round(boundaries[1])

    return df


def cities_unification(df):
    moscow_region = ['Aprelevka', 'Balashikha', 'Beloozyorskiy', 'Chekhov', 'Chernogolovka', 'Dedovsk',
                     'Dmitrov', 'Dolgoprudny', 'Domodedovo', 'Dubna', 'Dzerzhinsky', 'Elektrogorsk', 'Elektrostal',
                     'Elektrougli', 'Fryazino', 'Golitsyno', 'Istra', 'Ivanteyevka', 'Izhevsk', 'Kashira', 'Khimki',
                     'Khotkovo', 'Klin', 'Kolomna', 'Korolyov', 'Kotelniki', 'Krasnoarmeysk', 'Krasnogorsk',
                     'Krasnoznamensk', 'Kubinka', 'Kurovskoye', 'Likino-Dulyovo', 'Lobnya', 'Losino-Petrovsky',
                     'Lukhovitsy', 'Lytkarino', 'Lyubertsy', 'Mozhaysk', 'Mytishchi', 'Naro-Fominsk', 'Nakhabino',
                     'Noginsk', 'Odintsovo', 'Orekhovo-Zuyevo', 'Pavlovsky Posad', 'Podolsk', 'Protvino', 'Pushchino',
                     'Pushkino', 'Ramenskoye', 'Reutov', 'Ruza', 'Sergiyev Posad', 'Serpukhov', 'Solnechnogorsk',
                     'Staraya Kupavna', 'Stupino', 'Shchyolkovo', 'Shatura', 'Vidnoye', 'Volokolamsk', 'Voskresensk',
                     'Yakhroma', 'Zheleznodorozhny', 'Zhukovskiy', 'Zvenigorod', 'Moscow'
                     ]
    petersburg_region = ['Boksitogorsk', 'Volosovo', 'Volkhov', 'Vsevolozhsk', 'Vyborg', 'Vysotsk', 'Gatchina',
                         'Ivangorod', 'Kamennogorsk', 'Kingisepp', 'Kirishi', 'Kirovsk', 'Communar', 'Kudrovo',
                         'Lodeynoye Pole', 'Luban', 'Murino', 'Nikolskoye', 'Novaya Ladoga', 'Otradnoe', 'Pikalevo',
                         'Podporozhie', 'Primorsk', 'Priozersk', 'Svetogorsk', 'Sertolovo',
                         'Slantsy', 'Sosnovy Bor', 'Syasstroy', 'Tikhvin', 'Tosno', 'Shlisselburg', 'Petersburg'
                         ]
    df = df.copy()
    df['geo_city'] = df['geo_city'].replace(moscow_region, 'Moscow_region')
    df['geo_city'] = df['geo_city'].replace(petersburg_region, 'Petersburg_region')
    df['geo_city'] = df['geo_city'].replace(['(not set)'], 'Moscow_region')
    return df


def main():
    print('sber_auto_sub_prediction_pipeline')

    # Имена файлов для чтения
    # project_path = '~/PycharmProjects/pythonProject/Skillbox_diploma_proj/'
    project_path = 'C:/Users/GP62/PycharmProjects/pythonProject/Skillbox_diploma_proj/'
    file_path = os.path.join(project_path, 'data/raw data/')
    f_hits = 'ga_hits.csv'
    f_sessions = 'ga_sessions.csv'

    df_h = pd.read_csv(os.path.join(file_path, f_hits), usecols=['session_id', 'event_action'])
    df_s = pd.read_csv(os.path.join(file_path, f_sessions), dtype={'client_id': 'str'})

    X, y = merge_datasets(df_h, df_s)

    preprocessor1 = Pipeline(steps=[
        ('filter data', FunctionTransformer(filter_data)),
        ('imputer', FunctionTransformer(my_iputer)),
        ('remove_outliers', FunctionTransformer(remove_visit_number_outliers)),
        ('add time features', FunctionTransformer(add_time_features)),
        ('add binary features', FunctionTransformer(add_binary_features))

    ])

    preprocessor2 = Pipeline(steps=[
        #('imputer', SimpleImputer(strategy='constant', fill_value='other')),
        ('cities_unification', FunctionTransformer(cities_unification))
    ])

    categorical_transformer = Pipeline(steps=[
        ('target_encoder', TargetEncoder(smoothing=0.5, min_samples_leaf=2))
    ])

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor3 = ColumnTransformer(transformers=[
        ('num', numerical_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
        ('cat', categorical_transformer, make_column_selector(dtype_include=object))
    ], remainder='passthrough')

    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weights = dict(zip(classes, weights))

    models = [LogisticRegression(max_iter=2000, random_state=16),
              HistGradientBoostingClassifier(random_state=16),
              CatBoostClassifier(verbose=False, random_state=16),
              ]

    best_score = .0
    best_pipe = models[0]
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor_1', preprocessor1),
            ('preprocessor_2', preprocessor2),
            ('preprocessor_3', preprocessor3),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=StratifiedKFold(n_splits=5), scoring='roc_auc')
        auc = score.mean()
        print(f'model: {type(model).__name__}, auc_mean: {auc:.4f}')

        if auc > best_score:
            best_pipe = pipe
            best_score = auc
    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, roc_auc:  {best_score:.4f}')
    best_pipe.fit(X, y)

    pickle_file_name = 'auto_sub_prediction.pkl'
    file_path = os.path.join(project_path, 'data/models/', pickle_file_name)
    file_dict = {
        'model': best_pipe,
        'metadata': {
            'name': 'Auto subscribe prediction model',
            'author': 'Raul Gabbasov',
            'version': 1,
            'date': datetime.now(),
            'type': type(best_pipe.named_steps["classifier"]).__name__,
            'roc_auc': best_score,
            'class_weights': class_weights
        }
    }
    with open(file_path, 'wb') as file:
        dill.dump(file_dict, file)


if __name__ == '__main__':
    main()
