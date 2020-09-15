""" VAE class definition """

import torch

class VAE(torch.nn.Module):
    def __init__(self, embedding_dim = 64, hidden_dim = 128):
        """ Initialize the VAE forecasting model

        Attributes
        ----------
        embedding_dim : Embedding dimension of location coordinates
        hidden_dim : Dimension of hidden state of LSTM
        """
        super(VAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        ## LSTMs
        self.encoder1 = torch.nn.LSTMCell(self.embedding_dim, self.hidden_dim)
        self.encoder2 = torch.nn.LSTMCell(self.embedding_dim, self.hidden_dim)
        self.decoder = torch.nn.LSTMCell(2*self.hidden_dim, self.embedding_dim)

    def forward(self):
        """ Forward path for VAE forecasting model

        Attributes
        ----------
        """
        raise NotImplementedError
