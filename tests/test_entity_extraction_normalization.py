from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTITY_DIR = ROOT / "services" / "entity_extraction"
if str(ENTITY_DIR) not in sys.path:
    sys.path.insert(0, str(ENTITY_DIR))

if "pydantic" not in sys.modules:
    fake_pydantic = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **kwargs):
            annotations = getattr(self.__class__, "__annotations__", {})
            for name in annotations:
                if name in kwargs:
                    value = kwargs[name]
                else:
                    value = getattr(self.__class__, name, None)
                    if isinstance(value, list):
                        value = list(value)
                setattr(self, name, value)

    fake_pydantic.BaseModel = BaseModel
    sys.modules["pydantic"] = fake_pydantic

from extractor.entity_extractor import EntityExtractor  # noqa: E402
from extractor.models import Segment  # noqa: E402


class EntityExtractionNormalizationTest(unittest.TestCase):
    def test_deduplicates_and_normalizes_phones_and_emails(self) -> None:
        extractor = EntityExtractor(use_ner=False)
        entities = extractor.extract(
            [
                Segment(
                    start=0,
                    end=3,
                    speaker="spk_0",
                    text=(
                        "Мой номер телефона 8 (999) 123-45-67. "
                        "Повторяю номер телефона +7 999 123 45 67. "
                        "Почта TEST@example.com и еще раз test@example.com."
                    ),
                )
            ]
        )

        self.assertEqual(len(entities.phones), 1)
        self.assertEqual(entities.phones[0].value, "+79991234567")
        self.assertEqual(len(entities.emails), 1)
        self.assertEqual(entities.emails[0].value, "test@example.com")

    def test_normalizes_and_deduplicates_order_ids(self) -> None:
        extractor = EntityExtractor(use_ner=False)
        entities = extractor.extract(
            [
                Segment(
                    start=0,
                    end=2,
                    speaker="spk_1",
                    text=(
                        "Номер заказа 756.13.632. "
                        "Повторяю, заказ 75613632."
                    ),
                )
            ]
        )

        self.assertEqual(len(entities.order_ids), 1)
        self.assertEqual(entities.order_ids[0].value, "75613632")


if __name__ == "__main__":
    unittest.main()
