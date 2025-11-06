# Minimal placeholder “model”
class DummyModel:
    def __init__(self):
        self.version = "v0.1"

    def predict(self, x1: float, x2: float) -> float:
        # Simple linear rule for demo
        return 0.3 * x1 + 0.7 * x2

model = DummyModel()
