from matchpredictor.matchresults.result import Fixture, Outcome
from matchpredictor.predictors.predictor import Prediction, Predictor


class AlphabetPredictor(Predictor):
    def predict(self, fixture: Fixture) -> Prediction:
        # Predict based on the first team alphabetically winning
        return Prediction(Outcome.HOME if fixture.home_team.name < fixture.away_team.name else Outcome.AWAY)
