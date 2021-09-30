"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)
            
    
class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)
            
        
class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64], 
                 sparse=False, embedding_sharing=True):

        super().__init__()

        self.embedding_dim = embedding_dim

        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        
        self.embedding_sharing = embedding_sharing
        
        self.U = ScaledEmbedding(num_embeddings=num_users, embedding_dim=embedding_dim, sparse=sparse)
        self.Q = ScaledEmbedding(num_embeddings=num_items, embedding_dim=embedding_dim, sparse=sparse)
        self.A = ZeroEmbedding(num_embeddings=num_users, embedding_dim=1, sparse=sparse)
        self.B = ZeroEmbedding(num_embeddings=num_items, embedding_dim=1, sparse=sparse)

        if not self.embedding_sharing:
            self.U_reg = ScaledEmbedding(num_embeddings=num_users, embedding_dim=embedding_dim, sparse=sparse)
            self.Q_reg = ScaledEmbedding(num_embeddings=num_items, embedding_dim=embedding_dim, sparse=sparse)


        layers = []
        for in_d, out_d in zip(layer_sizes, layer_sizes[1:] + [1]):
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        
    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of 
            shape (batch,). This corresponds to p_ij in the 
            assignment.
        score: tensor
            Tensor of user-item score predictions of shape 
            (batch,). This corresponds to r_ij in the 
            assignment.
        """
        
        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************


        u = self.U(user_ids)
        q = self.Q(item_ids)
        a = self.A(user_ids)
        b = self.B(item_ids)

        predictions = torch.diag(u @ q.T).unsqueeze(1) + a + b

        if self.embedding_sharing:
            x = torch.cat([u, q, u * q], dim=1)
        else:
            u_reg = self.U_reg(user_ids)
            q_reg = self.Q_reg(item_ids)

            x = torch.cat([u_reg, q_reg, u_reg * q_reg], dim=1)

        score = self.model(x)

        predictions = predictions.squeeze()
        score = score.squeeze()

        #********************************************************
        #********************************************************
        #********************************************************
        return predictions, score
