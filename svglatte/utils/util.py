import pathlib
from abc import abstractmethod
from datetime import datetime

import torch


class Object(object):
    """
    Empty class for use as a default placeholder object.
    """
    pass


def get_str_formatted_time() -> str:
    """
    Returns the current time in the format of '%Y.%m.%d_%H.%M.%S'.

    Returns
    -------
    str
        The current time in the specified format
    """
    return datetime.now().strftime('%Y.%m.%d_%H.%M.%S')


HORSE = """               .,,.
             ,;;*;;;;,
            .-'``;-');;.
           /'  .-.  /*;;
         .'    \\d    \\;;               .;;;,
        / o      `    \\;    ,__.     ,;*;;;*;,
        \\__, _.__,'   \\_.-') __)--.;;;;;*;;;;,
         `""`;;;\\       /-')_) __)  `\' ';;;;;;
            ;*;;;        -') `)_)  |\\ |  ;;;;*;
            ;;;;|        `---`    O | | ;;*;;;
            *;*;\\|                 O  / ;;;;;*
           ;;;;;/|    .-------\\      / ;*;;;;;
          ;;;*;/ \\    |        '.   (`. ;;;*;;;
          ;;;;;'. ;   |          )   \\ | ;;;;;;
          ,;*;;;;\\/   |.        /   /` | ';;;*;
           ;latte/    |/       /   /__/   ';;;
           '*;;;/     |       /    |      ;*;
                `""""`        `""""`     ;'"""


def nice_print(msg, last=False):
    """
    Print a message in a nice format.

    Parameters
    ----------
    msg : str
        The message to be printed
    last : bool, optional
        Whether to print a blank line at the end, by default False

    Returns
    -------
    None
    """
    print()
    print("\033[0;35m" + msg + "\033[0m")
    if last:
        print()


class AttrDict(dict):
    """
    A dictionary class that can be accessed with attributes.
    Note that the dictionary keys must be strings
    and follow attribute naming rules to be accessible as attributes,
    e.g., the key "123xyz" will give a syntax error.

    Usage:
    ```
    x = AttrDict()
    x["jure"] = "mate"
    print(x.jure)
    # "mate"

    x[123] = "abc"
    x.123
    # SyntaxError: invalid syntax
    ```
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def ensure_dir(dirname):
    """
    Ensure that a directory exists. If it doesn't, create it.

    Parameters
    ----------
    dirname : str or pathlib.Path
        The directory to be ensured

    Returns
    -------
    None
    """
    dirname = pathlib.Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


class Embedder:
    """
    Abstract class for embedding input data.
    """

    @property
    @abstractmethod
    def out_dim(self):
        """
        The number of dimensions in the embedded data.

        Returns
        -------
        int
            The number of dimensions in the embedded data.
        """
        pass

    @abstractmethod
    def embed(self, x):
        """
        Embed the input data.

        Parameters
        ----------
        x : torch.Tensor of shape [..., ndim]
            The input data to be embedded

        Returns
        -------
        torch.Tensor
            The embedded data
        """
        pass

    @staticmethod
    def factory(embedding_style, **kwargs):
        """
        Factory method for creating instances of Embedder subclasses.

        Parameters
        ----------
        embedding_style : str
            The style of embedding to use, either "gaussian" or "nerf".
        kwargs :
            Keyword arguments passed to the corresponding subclass constructor.

        Returns
        -------
        Embedder
            An instance of a subclass of Embedder.
        """
        if embedding_style == "gaussian":
            return GaussianEmbedder(**kwargs)
        elif embedding_style == "nerf":
            return NerfEmbedder(**kwargs)
        else:
            raise RuntimeError("")


class GaussianEmbedder(Embedder):
    """
    A class for embedding 2D inputs to a higher dimensional space
    using frequencies randomly sampled from the Gaussian distribution.

    Parameters
    ----------
    input_dims : int, optional
        The dimensionality of the vectors to be embedded, e.g. 2 for 2D coordinates, 3 for 3D coordinates
    scale : int, optional
        The scale of the randomly sampled Gaussian distribution
    embedding_size : int, optional
        The size of the outputted embedding, i.e., the number

    Attributes
    ----------
    generator : torch.Generator
        The torch random number generator
    freq : torch.Tensor of shape (embedding_size, ndim)
        The embedding frequencies that were randomly sampled from the Gaussian distribution
    amplitude : torch.Tensor of shape (embedding_size,)
        The amplitude of the sinusoidal and cosinusoidal functions
    """

    def __init__(self, input_dims=2, scale=12, embedding_size=256, seed=72):
        self.embedding_size = embedding_size
        self.input_dims = input_dims
        self.scale = scale
        self.seed = seed

        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)
        self.freq = self.scale * torch.randn((self.embedding_size, self.input_dims), generator=self.generator)
        self.amplitude = torch.ones((self.freq.shape[0]))

        self._out_dim = embedding_size * 2

    def embed(self, x):
        return torch.concat([self.amplitude * torch.sin((2. * torch.pi * x) @ self.freq.T),
                             self.amplitude * torch.cos((2. * torch.pi * x) @ self.freq.T)], dim=-1)

    @property
    def out_dim(self):
        return self._out_dim


class NerfEmbedder(Embedder):
    """
    A class for embedding data using powers of two as frequencies.
    Modified from a PyTorch implementation of Neural Radiance Fields (NeRF):
    https://github.com/yenchenlin/nerf-pytorch/blob/63a5a630c9abd62b0f21c08703d0ac2ea7d4b9dd/run_nerf_helpers.py#L15

    Parameters
    ----------
    include_input : bool, optional
        Whether to include the input in the embedding
    input_dims : int, optional
        The dimensionality of the vectors to be embedded, e.g. 2 for 2D coordinates, 3 for 3D coordinates
    max_freq_log2 : int, optional
        The maximum frequency, given as log2 of the frequency, e.g. 10 would correspond to a max frequency of 2^10
    num_freqs : int, optional
        The number of frequencies to use
    log_sampling : bool, optional
        Whether to sample the frequencies logarithmically or uniformly, e.g. [1,2,4,8,16] or [1.0,4.75,8.5,12.25,16.0]
    periodic_fns : tuple, optional
        The periodic functions to use in the embedding (e.g. sin, cos)
    """

    def __init__(
            self,
            include_input=True,
            input_dims=2,
            num_freqs=14,
            max_freq_log2=14 - 1,
            log_sampling=True,
            periodic_fns=(torch.sin, torch.cos)
    ):
        self.include_input = include_input
        self.input_dims = input_dims
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns

        self._create_embedding_fn()

    def _create_embedding_fn(self):
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        if self.log_sampling:
            freq_bands = 2. ** torch.linspace(0., self.max_freq_log2, steps=self.num_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** self.max_freq_log2, steps=self.num_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self._embed_fns = embed_fns
        self._out_dim = out_dim

    def embed(self, x):
        return torch.cat([fn(x) for fn in self._embed_fns], -1)

    @property
    def out_dim(self):
        return self._out_dim


def resize_coordinate(x):
    """
    Resize input tensor from [0, 24] to [-1,+1].
    """
    assert x.max() <= 24
    assert x.min() >= -1
    return (x - 12) / 12


def identity_fn(x):
    return x


def embed_sequences(sequences, embedder: Embedder, resize_fn=identity_fn):
    """
    Embed the given sequences.

    Parameters
    ----------
    sequences : list of tensors
        List of tensors, where each tensor has shape (N, D),
        where N is the number of vectors in the sequence
        and D is the number of dimensions of each vector.
    embedder : Embedder
        An instance of the Embedder class, which will perform the embedding of the input points.
    resize_fn : callable, optional
        A function to apply to each input sequence before embedding.

    Returns
    -------
    list of tensors
        List of tensors, where each tensor has shape (N, embedder.out_dim), where N is the number of points in the
        sequence and out_dim is the number of dimensions in the embedded space.
    """
    embedded_sequences = [
        torch.concat([
            seq[:, :4],  # TODO hardcoded sequence positions
            embedder.embed(resize_fn(seq[:, 6:8])),  # TODO hardcoded sequence positions
            embedder.embed(resize_fn(seq[:, 4:6])),  # TODO hardcoded sequence positions
        ], dim=1)
        for seq in sequences
    ]
    return embedded_sequences


def pad_collate_fn(batch, pad_value, embedder: Embedder = None):
    """
    Function for padding sequences in a batch.

    Parameters
    ----------
    batch : List of tuples
        A list of tuples, each containing a sequence, image, and length.
    pad_value : float
        The value used for padding the sequences.
    embedder : Embedder, optional
        An instance of the Embedder class.

    Returns
    -------
    torch.Tensor, torch.Tensor, torch.Tensor
        The padded sequences, images, and lengths.
    """
    sequences, images, lengths = zip(*batch)
    if embedder is not None:
        sequences = embed_sequences(sequences, embedder, resize_coordinate)
    images = torch.cat(images).unsqueeze(1)
    lengths = torch.tensor(lengths)
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, padding_value=pad_value, batch_first=True)
    # padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, padding_value=pad_value)
    return padded_sequences, images, lengths
