import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid

DATA_FILE_URL = "https://atviriduomenys.vrk.lt/datasets/gov/vrk/Balsave/:format/csv"

BALSAI = "Balsave.csv"


def read_data() -> pd.DataFrame:
    data = pd.read_csv(BALSAI)
    data.drop(index=data.index[-1], axis=0, inplace=True)
    data["balsavusiu_dalis"] = data["atvyko_balsuoti"] / data["rinkeju_sarase"]
    return data


def prepare_rinkimai_data(data):
    X, y = (
        data[["apyl_id", "amziaus_grupe", "lytis"]],
        data["balsavusiu_dalis"],
    )
    return train_test_split(X, y, test_size=0.2)


def create_regression_model():
    cat_features = ["apyl_id", "lytis"]
    cat_transformer = make_pipeline(OneHotEncoder(handle_unknown="ignore"))
    num_features = ["amziaus_grupe"]
    num_transformer = make_pipeline(SimpleImputer(strategy="most_frequent"))
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
        ]
    )
    model = make_pipeline(
        preprocessor, PolynomialFeatures(degree=2), LinearRegression()
    )
    return model


def predict__voting_participation(
    model, vote_district: str, age_group: str, gender: str
):
    vote_district_dict = {"Romainių": 3666}
    data = {
        "apyl_id": vote_district_dict[vote_district],
        "amziaus_grupe": age_group,
        "lytis": gender,
    }
    z = pd.DataFrame([data])
    result = model.predict(z)
    return result


def create_testing_scenarios():
    param_grid = {"apyl_id": [3666], "amziaus_grupe": ["18-24", "80+"], "lytis": ["M"]}
    return pd.DataFrame(ParameterGrid(param_grid))


if __name__ == "__main__":
    vote_data = read_data()
    X_train, X_test, y_train, y_test = prepare_rinkimai_data(vote_data)
    model = create_regression_model()
    print(model)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    score = model.score(X_test, y_test)
    print(score)
    scenarios = create_testing_scenarios()
    print(scenarios)
    predictions = model.predict(scenarios)
    r = scenarios.assign(predictions=predictions)
    print(r)
