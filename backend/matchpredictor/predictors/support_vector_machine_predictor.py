from typing import List, Optional, Tuple
import numpy as np
from numpy import float64
from numpy.typing import NDArray
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from matchpredictor.matchresults.result import Fixture, Outcome, Result, Team
from matchpredictor.predictors.predictor import Predictor, Prediction

class SupportVectorMachinePredictor(Predictor):
    def __init__(self, model: SVC, team_encoding: OneHotEncoder) -> None:
        self.model = model
        self.team_encoding = team_encoding

    def predict(self, fixture: Fixture) -> Prediction:
        encoded_home_name = self.__encode_team(fixture.home_team)
        encoded_away_name = self.__encode_team(fixture.away_team)

        if encoded_home_name is None or encoded_away_name is None:
            return Prediction(outcome=Outcome.DRAW)

        x: NDArray[float64] = np.hstack([encoded_home_name, encoded_away_name])
        pred = self.model.predict(x)

        return Prediction(outcome=self.__map_prediction(pred))

    def __encode_team(self, team: Team) -> Optional[NDArray[float64]]:
        try:
            return self.team_encoding.transform(np.array(team.name).reshape(-1, 1)).toarray()
        except ValueError:
            return None

    def __map_prediction(self, pred: int) -> Outcome:
        if pred == 1:
            return Outcome.HOME
        elif pred == -1:
            return Outcome.AWAY
        else:
            return Outcome.DRAW

def build_svm_model(results: List[Result]) -> Tuple[SVC, OneHotEncoder]:
    team_names = np.array([r.fixture.home_team.name for r in results] + [r.fixture.away_team.name for r in results]).reshape(-1, 1)
    team_encoding = OneHotEncoder().fit(team_names)

    encoded_home_names = team_encoding.transform(np.array([r.fixture.home_team.name for r in results]).reshape(-1, 1)).toarray()
    encoded_away_names = team_encoding.transform(np.array([r.fixture.away_team.name for r in results]).reshape(-1, 1)).toarray()

    x: NDArray[float64] = np.hstack([encoded_home_names, encoded_away_names])
    y = np.sign(np.array([r.home_goals - r.away_goals for r in results]))

    model = SVC(kernel="linear")
    model.fit(x, y)

    return model, team_encoding

def train_svm_predictor(results: List[Result]) -> Predictor:
    model, team_encoding = build_svm_model(results)
    return SupportVectorMachinePredictor(model, team_encoding)
