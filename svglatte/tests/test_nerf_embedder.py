# TODO refactor the location of tests

import unittest

import torch
from absl.testing import parameterized

from svglatte.utils.util import NerfEmbedder


class TestNerfEmbedder(parameterized.TestCase):
    @parameterized.product(
        input_dims=[1, 2, 4],
        include_input=[True, False],
        max_freq_log2=[30, 14, 3],
        num_freqs=[14, 4],
        log_sampling=[True, False],
        periodic_fns=[(torch.sin, torch.cos), (torch.sin,), (torch.relu, torch.sin)]
    )
    def test_create_embedding_fn(self, input_dims, include_input, max_freq_log2, num_freqs, log_sampling, periodic_fns):
        embedder = NerfEmbedder(
            include_input=include_input,
            input_dims=input_dims,
            num_freqs=num_freqs,
            max_freq_log2=max_freq_log2,
            log_sampling=log_sampling,
            periodic_fns=periodic_fns,
        )

        assert embedder.include_input == include_input
        assert embedder.input_dims == input_dims
        assert embedder.max_freq_log2 == max_freq_log2
        assert embedder.num_freqs == num_freqs
        assert embedder.log_sampling == log_sampling
        assert embedder.periodic_fns == periodic_fns
        assert embedder.out_dim == input_dims * include_input + input_dims * num_freqs * len(periodic_fns)

    @parameterized.product(
        input_dims=[1, 2, 4],
        include_input=[True, False],
        max_freq_log2=[30, 14, 3],
        num_freqs=[14, 4],
        log_sampling=[True, False],
        periodic_fns=[(torch.sin, torch.cos), (torch.sin,), (torch.relu, torch.sin)]
    )
    def test_embed(self, input_dims, include_input, max_freq_log2, num_freqs, log_sampling, periodic_fns):
        embedder = NerfEmbedder(
            include_input=include_input,
            input_dims=input_dims,
            num_freqs=num_freqs,
            max_freq_log2=max_freq_log2,
            log_sampling=log_sampling,
            periodic_fns=periodic_fns,
        )

        inputs = torch.randn(100, input_dims)
        embeddings = embedder.embed(inputs)
        assert embeddings.shape == (100, embedder.out_dim)

        inputs_2 = torch.randn(input_dims)
        embeddings_2 = embedder.embed(inputs_2)
        assert embeddings_2.shape == (embedder.out_dim,)

    def test_for_expected_embedding_1(self):
        periodic_fns = [torch.sin, torch.cos]
        embedder = NerfEmbedder(
            include_input=True,
            input_dims=3,
            num_freqs=4,
            max_freq_log2=3,
            log_sampling=True,
            periodic_fns=periodic_fns,
        )
        inputs = torch.tensor([[0.15, 0.5, 0.72], [-1, 0, 1]])
        embeddings = embedder.embed(inputs)

        freq_bands = torch.tensor([1., 2., 4., 8.])
        embedding_parts_list = [inputs]
        for freq in freq_bands:
            for fn in periodic_fns:
                embedding_parts_list += [fn(inputs * freq)]
                # for i in inputs:
                #     for j in i:
                #         print(f"{j:.4f} --> {j * freq:.4f} --> {fn(j):.4f} [freq={freq}] [fn={fn}]")
        expected_embedding = torch.concat(embedding_parts_list, dim=-1)
        assert torch.allclose(embeddings, expected_embedding)

    def test_for_expected_embedding_2(self):
        embedder = NerfEmbedder(
            include_input=True,
            input_dims=3,
            num_freqs=4,
            max_freq_log2=3,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        inputs = torch.tensor([[0.15, 0.5, 0.72], [-1, 0, 1]])
        embeddings = embedder.embed(inputs)

        # print(embeddings.tolist())
        # sed -i "s/,/,\n/3;P;D" tmp.txt
        expected_embeddings = torch.tensor(
            [[0.15000000596046448, 0.5, 0.7200000286102295,
              0.14943814277648926, 0.4794255495071411, 0.6593846678733826,
              0.9887710809707642, 0.8775825500488281, 0.7518057227134705,
              0.29552021622657776, 0.8414709568023682, 0.9914583563804626,
              0.9553365111351013, 0.5403023362159729, 0.1304236501455307,
              0.5646424889564514, 0.9092974066734314, 0.2586192488670349,
              0.8253356218338013, -0.416146844625473, -0.9659793376922607,
              0.9320390820503235, -0.756802499294281, -0.49964168667793274,
              0.3623577058315277, -0.6536436080932617, 0.86623215675354],
             [-1.0, 0.0, 1.0,
              -0.8414709568023682, 0.0, 0.8414709568023682, 0.5403023362159729, 1.0, 0.5403023362159729,
              -0.9092974066734314, 0.0, 0.9092974066734314, -0.416146844625473, 1.0, -0.416146844625473,
              0.756802499294281, 0.0, -0.756802499294281, -0.6536436080932617, 1.0, -0.6536436080932617,
              -0.9893582463264465, 0.0, 0.9893582463264465, -0.1455000340938568, 1.0, -0.1455000340938568]]
        )
        assert torch.allclose(embeddings, expected_embeddings)


if __name__ == '__main__':
    unittest.main()
