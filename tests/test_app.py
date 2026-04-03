import io
import unittest

import pandas as pd
from fastapi.testclient import TestClient

from app import app


class TestFastAPIApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.test_df = pd.read_csv("data/testing.csv").head(3)

    def test_home_page_loads(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Urban Land Cover Predictor", response.text)

    def test_predict_csv_upload_returns_results(self):
        csv_buffer = io.StringIO()
        self.test_df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode("utf-8")

        response = self.client.post(
            "/predict",
            files={"file": ("sample.csv", csv_bytes, "text/csv")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Prediction Results", response.text)
        self.assertIn("prediction", response.text)


if __name__ == "__main__":
    unittest.main()
