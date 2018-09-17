import sys
sys.path.append("..")
from read_data import get_data

train_data, train_predictions, test_data, test_predictions = get_data()

n_features = train_data.shape[0]

random_weights_ = (np.sqrt(2 * self.gamma) * random_state.normal(
            size=(n_features, self.n_components)))
