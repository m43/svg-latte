# TODO refactor the location of tests

import unittest

import torch
from absl.testing import parameterized

from svglatte.utils.util import GaussianEmbedder


class TestGaussianEmbedder(parameterized.TestCase):
    @parameterized.product(
        input_dims=[1, 2, 4],
        scale=[1, 10],
        embedding_size=[1, 30, 256],
        seed=[72, 36],
    )
    def test_embed(self, input_dims, scale, embedding_size, seed):
        embedder = GaussianEmbedder(
            input_dims=input_dims,
            scale=scale,
            embedding_size=embedding_size,
            seed=seed,
        )

        inputs = torch.randn(100, input_dims)
        embeddings = embedder.embed(inputs)
        assert embeddings.shape == (100, embedder.out_dim)

        inputs_2 = torch.randn(input_dims)
        embeddings_2 = embedder.embed(inputs_2)
        assert embeddings_2.shape == (embedder.out_dim,)

    def test_for_expected_embedding(self):
        embedder = GaussianEmbedder(
            input_dims=3,
            scale=10,
            embedding_size=10,
            seed=72,
        )
        inputs = torch.tensor([[0, 0.5, 1], [-1, 0, 1]])
        embeddings = embedder.embed(inputs)
        
        expected_embeddings = torch.tensor(
            [[-0.6632360816001892, -0.9538021683692932, -0.9303285479545593, 0.7226450443267822, 0.7425926327705383,
              -0.4597611725330353, 0.3829289674758911, -0.018863791599869728, 0.9841323494911194, 0.5045086145401001,
              -0.7484102845191956, -0.3004353642463684, 0.36672714352607727, 0.6912193298339844, -0.6697433590888977,
              -0.8880426287651062, 0.9237778186798096, -0.9998220801353455, -0.17743586003780365, -0.8634066581726074],
             [-0.5181511640548706, -0.9389735460281372, -0.3012053966522217, 0.5573055148124695, 0.020015254616737366,
              0.9685661196708679, -0.809158444404602, -0.9073007106781006, 0.997526228427887, -0.8431944251060486,
              -0.8552890419960022, -0.34398943185806274, 0.9535592794418335, -0.8303074836730957, 0.9997996687889099,
              -0.2487562596797943, -0.5875904560089111, -0.420482337474823, -0.07029476016759872, -0.5376087427139282]])
        assert torch.equal(embeddings, expected_embeddings)


if __name__ == '__main__':
    unittest.main()
