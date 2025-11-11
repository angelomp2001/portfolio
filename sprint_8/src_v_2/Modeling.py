

class Model:
    def __init__(self,features, target, split_ratio=(0.6,0.2,0.2), random_state=12345):
        self.data = features
        self.target = target
        self.split_ratio = split_ratio
        self.random_state = random_state
        self.df = None
        self.scores = None

    def fit(self, model = dict = None, target_threshold = 0.5):
        self.model = model
        self.target_threshold = target_threshold

        for model_type, models in model_options.items():
        print(f'model type: {model_type}')

        

