import numpy as np
import pandas as pd
from thermal_benchmark.data import FEATURES, TEMPERATURES, load_data, split_data
from thermal_benchmark.evaluation import recursive_rollout, temperature_noise
from thermal_benchmark.models import Persistence, TorchMLP


def test_chronological_targets_do_not_cross_boundaries():
    df = load_data("data/house_data.csv")
    for days in (3, 5, 10):
        train, valid, test = split_data(df, days)
        assert len(train[0]) == 24*days-1
        assert len(valid[0]) == 47 and len(test[0]) == 96
        np.testing.assert_array_equal(train[1][-1], df.iloc[239][TEMPERATURES].to_numpy(float))
        np.testing.assert_array_equal(valid[0][0, :9], df.iloc[240][TEMPERATURES].to_numpy(float))
        np.testing.assert_array_equal(test[0][0, :9], df.iloc[288][TEMPERATURES].to_numpy(float))


def test_recursive_forecast_uses_predictions_and_aligned_forcing():
    a = np.zeros((30, 20))
    a[:, :9] = np.arange(30)[:, None]*1000  # Must not be read after initialization.
    a[:, 9] = np.arange(30)
    df = pd.DataFrame(a, columns=FEATURES)
    class AddForcing:
        def predict(self, x):
            return x[:, :9]+x[:, 9:10]+1
    pred, truth = recursive_rollout(AddForcing(), df)
    np.testing.assert_array_equal(pred[0, :, 0], np.cumsum(np.arange(24)+1))
    assert pred.shape == (6, 24, 9)
    assert truth[0, 0, 0] == 1000


def test_persistence_and_noise_are_paired():
    x = np.arange(200, dtype=float).reshape(10, 20)
    a = temperature_noise(x, .5, 1)
    np.testing.assert_array_equal(a, temperature_noise(x, .5, 1))
    np.testing.assert_array_equal(a[:, 9:], x[:, 9:])
    np.testing.assert_array_equal(Persistence().predict(a), a[:, :9])
    np.testing.assert_array_equal(temperature_noise(x, 0., 1), x)


def test_scalers_are_fit_on_training_only():
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=(30, 20)), rng.normal(size=(30, 9))
    model = TorchMLP(epochs=10).fit(x, y, x+100, y+100)
    np.testing.assert_allclose(model.x_scaler.mean_, x.mean(0))
    np.testing.assert_allclose(model.y_scaler.mean_, y.mean(0))
    assert np.isfinite(model.predict(x)).all()
