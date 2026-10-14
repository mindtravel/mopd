import unittest

import torch

from mopd.data.templates import apply_chat_template_ids


class ListTokenizer:
    def apply_chat_template(self, messages, *, tokenize, return_tensors=None, **kwargs):
        self.args = (tokenize, return_tensors, kwargs)
        return [11, 12, 13]


class EncodingTokenizer:
    def apply_chat_template(self, messages, *, tokenize, return_tensors=None, **kwargs):
        self.args = (tokenize, return_tensors, kwargs)
        if return_tensors:
            return {"input_ids": torch.tensor([[11, 12, 13]])}
        return {"input_ids": [11, 12, 13], "attention_mask": [1, 1, 1]}


class ChatTemplateIdsTests(unittest.TestCase):
    def test_normalizes_legacy_list_return(self):
        result = apply_chat_template_ids(ListTokenizer(), [{"role": "user", "content": "x"}])
        self.assertEqual(result, [11, 12, 13])

    def test_extracts_batch_encoding_input_ids(self):
        result = apply_chat_template_ids(EncodingTokenizer(), [{"role": "user", "content": "x"}])
        self.assertEqual(result, [11, 12, 13])

    def test_preserves_requested_tensor_return(self):
        result = apply_chat_template_ids(
            EncodingTokenizer(), [{"role": "user", "content": "x"}], return_tensors="pt"
        )
        self.assertTrue(torch.equal(result, torch.tensor([[11, 12, 13]])))


if __name__ == "__main__":
    unittest.main()
