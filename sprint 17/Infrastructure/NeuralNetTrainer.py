from libraries import *

# from infrastructure.NeuralNetTrainer import NeuralNetTrainer 
class NeuralNetTrainer:
    def __init__(self, features, k_folds: int, drop_rate: float, random_state: int = 12345):
        self.features = features
        self.k_folds = k_folds
        self.drop_rate = drop_rate
        self.random_state = random_state
        self.kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=12345)
        tf.random.set_seed(random_state)

    def evaluate(self, target, preprocessor):
        train_scores = []
        test_scores = []

        # cross val score
        for train_idx, test_idx in self.kf.split(self.features):
            feature_train, feature_test = self.features.iloc[train_idx], self.features.iloc[test_idx]
            target_train, target_test = target.iloc[train_idx], target.iloc[test_idx]

            # Preprocess
            X_train = preprocessor.fit_transform(feature_train)
            X_test = preprocessor.transform(feature_test)

            # Build new model for each fold
            nn_model = Sequential([
                Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
                Dropout(self.drop_rate),
                Dense(256, activation='relu'),
                Dropout(self.drop_rate),
                Dense(128, activation='relu'),
                Dense(1, activation='sigmoid')
            ])

            nn_model.compile(optimizer=Adam(learning_rate=0.001),
                            loss='binary_crossentropy',
                            metrics=['AUC'])

            # Fit model
            nn_model.fit(X_train, target_train, epochs=30, batch_size=32, verbose=0)

            # Calc Predictions
            y_pred_train = nn_model.predict(X_train).ravel()
            y_pred_test = nn_model.predict(X_test).ravel()

            # Calc AUC
            train_auc = roc_auc_score(target_train, y_pred_train)
            test_auc = roc_auc_score(target_test, y_pred_test)
            train_scores.append(train_auc)
            test_scores.append(test_auc)

        return {'NeuralNetwork': {
            'Train AUC': np.mean(train_scores),
            'Test AUC': np.mean(test_scores)
        }}
