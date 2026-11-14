
import unittest

from rag.structure_chunker import pack_units


class TestPackUnits(unittest.TestCase):

    def test_normal_packing(self):
        """测试正常文本组合。"""
        units = [
            "A" * 200,
            "B" * 150,
            "C" * 250,
            "D" * 100,
        ]

        result = pack_units(
            units,
            chunk_size=500,
            chunk_overlap=100,
        )

        expected = [
            units[0] + "\n\n" + units[1],
            units[2] + "\n\n" + units[3],
        ]

        self.assertEqual(result, expected)

    def test_overlap(self):
        """测试完整Unit重叠。"""
        units = [
            "A" * 200,
            "B" * 80,
            "C" * 250,
            "D" * 100,
        ]

        result = pack_units(
            units,
            chunk_size=500,
            chunk_overlap=100,
        )

        expected = [
            units[0] + "\n\n" + units[1],
            units[1] + "\n\n" + units[2]
            + "\n\n" + units[3],
        ]

        self.assertEqual(result, expected)

    def test_chunk_length_limit(self):
        """测试所有Chunk都满足长度上限。"""
        units = [
            "A" * 120,
            "B" * 90,
            "C" * 180,
            "D" * 110,
            "E" * 150,
        ]

        result = pack_units(
            units,
            chunk_size=300,
            chunk_overlap=100,
        )

        self.assertTrue(
            all(len(chunk) <= 300 for chunk in result)
        )

    def test_oversized_unit(self):
        """测试超长Unit必须触发异常。"""
        units = ["A" * 800]

        with self.assertRaises(ValueError):
            pack_units(
                units,
                chunk_size=500,
                chunk_overlap=100,
            )

    def test_empty_input(self):
        """测试空输入。"""
        result = pack_units(
            [],
            chunk_size=500,
            chunk_overlap=100,
        )

        self.assertEqual(result, [])

    def test_invalid_parameters(self):
        """测试非法参数。"""
        with self.assertRaises(ValueError):
            pack_units(
                ["Hello"],
                chunk_size=500,
                chunk_overlap=500,
            )

    def test_all_units_preserved(self):
        """测试每个Unit至少出现一次。"""
        units = [
            "Unique paragraph alpha.",
            "Unique paragraph beta.",
            "Unique paragraph gamma.",
            "Unique paragraph delta.",
        ]

        result = pack_units(
            units,
            chunk_size=60,
            chunk_overlap=20,
        )

        combined = "\n".join(result)

        for unit in units:
            self.assertIn(unit, combined)


if __name__ == "__main__":
    unittest.main()
"""
四、运行测试
在项目根目录执行：
python -m unittest discover -s tests -v


预期能够看到类似：
test_all_units_preserved ... ok
test_chunk_length_limit ... ok
test_empty_input ... ok
test_invalid_parameters ... ok
test_normal_packing ... ok
test_overlap ... ok
test_oversized_unit ... ok

----------------------------------------------------------------------
Ran 7 tests

OK


这里解释两个参数：
- discover：自动发现测试文件。
- -s tests：指定从 tests 目录开始查找。
- -v：Verbose，显示每个测试的执行结果。
如果失败，会显示失败的测试名称和断言差异。

这也是你应该逐渐养成的习惯：修改核心算法后先运行测试，而不是直接重新处理整篇 PDF。
"""