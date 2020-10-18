""" VAE class definition """
import itertools
import collections

import torch

from .modules import InputEmbedding, Hidden2Normal

import trajnetplusplustools
from ..augmentation import inverse_scene
from .utils import center_scene, sample_multivariate_distribution
import logging

NAN = float('nan')

class VAE(torch.nn.Module):
    """VAE forecasting model
    """
    def __init__(self, embedding_dim=64, hidden_dim=128, \
        latent_dim=128, pool=None, pool_to_input=True, \
            goal_dim=None, goal_flag=False, num_modes=1, debug_mode = False):
        """ Initialize the VAE forecasting model

        Attributes
        ----------
        embedding_dim : Embedding dimension of location coordinates
        hidden_dim : Dimension of hidden state of LSTM
        pool : interaction module
        pool_to_input : Bool
            if True, the interaction vector is concatenated to the input embedding of LSTM [preferred]
            if False, the interaction vector is added to the LSTM hidden-state
        goal_dim : Embedding dimension of the unit vector pointing towards the goal
        goal_flag: Bool
            if True, the embedded goal vector is concatenated to the input embedding of LSTM 
        """
        super(VAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        vae_input_dim = torch.sqrt(torch.Tensor([2*self.hidden_dim])).item()
        self.vae_input_dims = (int(vae_input_dim), int(vae_input_dim))
        self.pool = pool
        self.pool_to_input = pool_to_input
        self.num_modes = num_modes
        self.debug_mode = debug_mode

        ## Location
        scale = 4.0
        self.input_embedding = InputEmbedding(2, self.embedding_dim, scale)

        ## Goal
        self.goal_flag = goal_flag
        self.goal_dim = goal_dim or embedding_dim
        self.goal_embedding = InputEmbedding(2, self.goal_dim, scale)
        goal_rep_dim = self.goal_dim if self.goal_flag else 0

        ## Pooling
        pooling_dim = 0
        if pool is not None and self.pool_to_input:
            pooling_dim = self.pool.out_dim

        ## LSTM encoders
        self.obs_encoder = torch.nn.LSTMCell(self.embedding_dim + goal_rep_dim + pooling_dim, self.hidden_dim)
        self.pre_encoder = torch.nn.LSTMCell(self.embedding_dim + goal_rep_dim + pooling_dim, self.hidden_dim)

        ## cVAE
        self.vae_encoder = VAEEncoder(self.vae_input_dims, 2*self.latent_dim)
        self.vae_decoder = VAEDecoder(self.latent_dim, self.hidden_dim)

        ## LSTM decoder
        self.decoder = torch.nn.LSTMCell(self.embedding_dim + goal_rep_dim + pooling_dim, self.hidden_dim)

        # Predict the parameters of a multivariate normal:
        # mu_vel_x, mu_vel_y, sigma_vel_x, sigma_vel_y, rho
        self.hidden2normal_encoder = Hidden2Normal(self.hidden_dim)
        self.hidden2normal_decoder = Hidden2Normal(2*self.hidden_dim)

        assert(vae_input_dim.is_integer())

        # Logger
        # # configure logging
        if self.debug_mode:
            from pythonjsonlogger import jsonlogger
            import sys
            handler = logging.FileHandler('z_val.log')   
            formatter = logging.Formatter('%(message)s')     
            handler.setFormatter(formatter)

            self.logger = logging.getLogger('z_val')
            self.logger.setLevel(logging.DEBUG)
            self.logger.addHandler(handler)
            self.logger.debug({
                'type': 'process',
                'argv': sys.argv,
            }) 

        

    def step(self, lstm, hidden_cell_state, obs1, obs2, goals, batch_split):
        """Do one step of prediction: two inputs to one normal prediction.
        
        Parameters
        ----------
        lstm: torch nn module [Encoder / Decoder]
            The module responsible for prediction
        hidden_cell_state : tuple (hidden_state, cell_state)
            Current hidden_cell_state of the pedestrians
        obs1 : Tensor [num_tracks, 2]
            Previous x-y positions of the pedestrians
        obs2 : Tensor [num_tracks, 2]
            Current x-y positions of the pedestrians
        goals : Tensor [num_tracks, 2]
            Goal coordinates of the pedestrians
        
        Returns
        -------
        hidden_cell_state : tuple (hidden_state, cell_state)
            Updated hidden_cell_state of the pedestrians
        normals : Tensor [num_tracks, 5]
            Parameters of a multivariate normal of the predicted position 
            with respect to the current position
        """
        num_tracks = len(obs2)
        # mask for pedestrians absent from scene (partial trajectories)
        # consider only the hidden states of pedestrains present in scene
        track_mask = (torch.isnan(obs1[:, 0]) + torch.isnan(obs2[:, 0])) == 0

        ## Masked Hidden Cell State
        hidden_cell_stacked = [
            torch.stack([h for m, h in zip(track_mask, hidden_cell_state[0]) if m], dim=0),
            torch.stack([c for m, c in zip(track_mask, hidden_cell_state[1]) if m], dim=0),
        ]

        ## Mask current velocity & embed
        curr_velocity = obs2 - obs1
        curr_velocity = curr_velocity[track_mask]
        input_emb = self.input_embedding(curr_velocity)

        ## Mask Goal direction & embed
        if self.goal_flag:
            ## Get relative direction to goals (wrt current position)
            norm_factors = (torch.norm(obs2 - goals, dim=1))
            goal_direction = (obs2 - goals) / norm_factors.unsqueeze(1)
            goal_direction[norm_factors == 0] = torch.Tensor([0., 0.], device=obs1.device)
            goal_direction = goal_direction[track_mask]
            goal_emb = self.goal_embedding(goal_direction)
            input_emb = torch.cat([input_emb, goal_emb], dim=1)

        ## Mask & Pool per scene
        if self.pool is not None:
            hidden_states_to_pool = torch.stack(hidden_cell_state[0]).clone() # detach?
            batch_pool = []
            ## Iterate over scenes
            for (start, end) in zip(batch_split[:-1], batch_split[1:]):
                ## Mask for the scene
                scene_track_mask = track_mask[start:end]
                ## Get observations and hidden-state for the scene
                prev_position = obs1[start:end][scene_track_mask]
                curr_position = obs2[start:end][scene_track_mask]
                curr_hidden_state = hidden_states_to_pool[start:end][scene_track_mask]

                # LSTM-Based Interaction Encoders. Provide track_mask to the interaction encoder LSTMs
                if self.pool.__class__.__name__ in {'NN_LSTM', 'TrajectronPooling', 'SAttention_fast'}:
                    ## Everyone absent by default
                    interaction_track_mask = torch.zeros(num_tracks, device=obs1.device).bool()
                    ## Only those visible in current scene are present
                    interaction_track_mask[start:end] = track_mask[start:end]
                    self.pool.track_mask = interaction_track_mask

                pool_sample = self.pool(curr_hidden_state, prev_position, curr_position)
                batch_pool.append(pool_sample)

            pooled = torch.cat(batch_pool)
            if self.pool_to_input:
                input_emb = torch.cat([input_emb, pooled], dim=1)
            else:
                hidden_cell_stacked[0] += pooled

        # LSTM step
        hidden_cell_stacked = lstm(input_emb, hidden_cell_stacked)
        normal_masked = self.hidden2normal_encoder(hidden_cell_stacked[0]) if hidden_cell_state[0][0].size(0) == self.hidden_dim \
                        else self.hidden2normal_decoder(hidden_cell_stacked[0])

        # unmask [Update hidden-states and next velocities of pedestrians]
        normal = torch.full((track_mask.size(0), 5), NAN, device=obs1.device)
        mask_index = [i for i, m in enumerate(track_mask) if m]
        for i, h, c, n in zip(mask_index,
                              hidden_cell_stacked[0],
                              hidden_cell_stacked[1],
                              normal_masked):
            hidden_cell_state[0][i] = h
            hidden_cell_state[1][i] = c
            normal[i] = n

        return hidden_cell_state, normal

    def forward(self, observed, goals, batch_split, prediction_truth=None, n_predict=None):
        """Forecast the entire sequence
        
        Parameters
        ----------
        observed : Tensor [obs_length, num_tracks, 2]
            Observed sequences of x-y coordinates of the pedestrians
        goals : Tensor [num_tracks, 2]
            Goal coordinates of the pedestrians
        batch_split : Tensor [batch_size + 1]
            Tensor defining the split of the batch.
            Required to identify the tracks of to the same scene        
        prediction_truth : Tensor [pred_length - 1, num_tracks, 2]
            Prediction sequences of x-y coordinates of the pedestrians
            Helps in teacher forcing wrt neighbours positions during training
        n_predict: Int
            Length of sequence to be predicted during test time

        Returns
        -------
        rel_pred_scene : Tensor [pred_length, num_tracks, 5]
            Predicted velocities of pedestrians as multivariate normal
            i.e. positions relative to previous positions
        pred_scene : Tensor [pred_length, num_tracks, 2]
            Predicted positions of pedestrians i.e. absolute positions
        """
        assert ((prediction_truth is None) + (n_predict is None)) == 1
        if n_predict is not None:
            # -1 because one prediction is done by the encoder already
            prediction_truth = [None for _ in range(n_predict - 1)]

        # initialize: Because of tracks with different lengths and the masked
        # update, the hidden state for every LSTM needs to be a separate object
        # in the backprop graph. Therefore: list of hidden states instead of
        # a single higher rank Tensor.
        num_tracks = observed.size(1)
        # hidden cell state for observer encoder 
        hidden_cell_state_obs = (
            [torch.zeros(self.hidden_dim, device=observed.device) for _ in range(num_tracks)],
            [torch.zeros(self.hidden_dim, device=observed.device) for _ in range(num_tracks)],
        )
        # hidden cell state for prediction encoder
        hidden_cell_state_pre = (
            [torch.zeros(self.hidden_dim, device=observed.device) for _ in range(num_tracks)],
            [torch.zeros(self.hidden_dim, device=observed.device) for _ in range(num_tracks)],
        )

        ## LSTM-Based Interaction Encoders. Initialze Hidden state
        if self.pool.__class__.__name__ in {'NN_LSTM', 'TrajectronPooling', 'SAttention', 'SAttention_fast'}:
            self.pool.reset(num_tracks, device=observed.device)


        if len(observed) == 2:
            positions = [observed[-1]]

        # list of predictions store a dictionary. Each key corresponds to one mode
        normals = {mode: [] for mode in range(self.num_modes)} # predicted normal parameters for both phases
        positions = {mode: [] for mode in range(self.num_modes)} # true (during obs phase) and predicted positions

        ## Observer encoder
        for obs1, obs2 in zip(observed[:-1], observed[1:]):
            # LSTM Step
            hidden_cell_state_obs, normal = self.step(self.obs_encoder, hidden_cell_state_obs, obs1, obs2, goals, batch_split)
            # concat predictions
            
            for mode_n, mode_p in zip(normals.keys(), positions.keys()):
                normals[mode_n].append(normal)
                positions[mode_p].append(obs2 + normal[:, :2]) # no sampling, just mean
    
        # initialize predictions with last position to form velocity
        prediction_truth = list(itertools.chain.from_iterable(
            (observed[-1:], prediction_truth)
        ))
        
        ## Prediction encoder
        if self.training:
            for obs1, obs2 in zip(prediction_truth[:-1], prediction_truth[1:]):
                # LSTM Step
                hidden_cell_state_pre, _ = self.step(self.pre_encoder, hidden_cell_state_pre, obs1, obs2, goals, batch_split)

            # Concatenation of hidden states
            hidden_cell_state = tuple([[torch.cat((track_obs, track_pre), dim=0) for track_obs, track_pre in zip(obs, pre)] for obs, pre in zip(hidden_cell_state_obs, hidden_cell_state_pre)])
        
        else: # eval mode 
            hidden_cell_state = hidden_cell_state_obs
        
        # Save hidden cell state for multiple predictions
        saved_hidden_cell_state = hidden_cell_state

        ## VAE encoder, latent distribution
        if self.training:
            hidden_state = hidden_cell_state[0]
            z_mu, z_var_log = self.vae_encoder(hidden_state)
            z_distr = torch.cat((z_mu, z_var_log), dim=1)
        else: 
            z_distr = None

        # Make k predictions
        for k in range(self.num_modes):
            hidden_cell_state = saved_hidden_cell_state
            if self.training:
                ## Sampling using "reparametrization trick"
                # See Kingma & Wellig, Auto-Encoding Variational Bayes, 2014 (arXiv:1312.6114)
                epsilon = torch.empty(size=z_mu.size()).normal_(mean=0, std=1)
                z_val = z_mu + torch.exp(z_var_log/2) * epsilon
                if self.debug_mode:  # TODO: remove
                    import numpy as np
                    if(np.random.random() > 0.9):
                        self.logger.debug({
                            "z_val": list(z_val[0].detach().numpy()),
                        })
            
            else: # eval mode
                # Draw a sample from the learned multivariate distribution (z_mu, z_var_log)
                z_mu = torch.zeros(self.latent_dim)
                z_var_log = torch.ones(self.latent_dim)
                z_val = sample_multivariate_distribution(z_mu, z_var_log)

            ## VAE decoder
            x_reconstr = self.vae_decoder(z_val)
            hidden_cell = [hidden_cell_state_obs[0][i] * x_reconstr[i] for i in range(num_tracks)]
            hidden_cell_state = (hidden_cell, hidden_cell_state_obs[1])
            ## decoder, predictions
            for obs1, obs2 in zip(prediction_truth[:-1], prediction_truth[1:]):
                if obs1 is None:
                    obs1 = positions[k][-2].detach()  # DETACH!!!
                else:
                    for primary_id in batch_split[:-1]:
                        obs1[primary_id] = positions[k][-2][primary_id].detach()  # DETACH!!!
                if obs2 is None:
                    obs2 = positions[k][-1].detach()
                else:
                    for primary_id in batch_split[:-1]:
                        obs2[primary_id] = positions[k][-1][primary_id].detach()  # DETACH!!!
                hidden_cell_state, normal = self.step(self.decoder, hidden_cell_state, obs1, obs2, goals, batch_split)

                # concat predictions
                normals[k].append(normal)
                positions[k].append(obs2 + normal[:, :2])  # no sampling, just mean

        # Pred_scene: Tensor [seq_length, num_tracks, 2]
        #    Absolute positions of all pedestrians
        # Rel_pred_scene: Tensor [seq_length, num_tracks, 5]
        #    Velocities of all pedestrians

        rel_pred_scene = [torch.stack(normals[mode_n], dim=0) for mode_n in normals.keys()]
        pred_scene = [torch.stack(positions[mode_p], dim=0) for mode_p in positions.keys()]

        return rel_pred_scene, pred_scene, z_distr


class VAEPredictor(object):
    def __init__(self, model):
        super(VAEPredictor, self).__init__()
        self.model = model

    def save(self, state, filename):
        with open(filename, 'wb') as f:
            torch.save(self, f)

        # # during development, good for compatibility across API changes:
        # # Save state for optimizer to continue training in future
        with open(filename + '.state', 'wb') as f:
            torch.save(state, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return torch.load(f)


    def __call__(self, paths, scene_goal, n_predict=12, predict_all=True, obs_length=9, start_length=0, modes=1, args=None):
        self.model.eval()
        # self.model.train()
        with torch.no_grad():
            xy = trajnetplusplustools.Reader.paths_to_xy(paths)
            batch_split = [0, xy.shape[1]]

            ## Drop Distant (for real data)
            # xy, mask = drop_distant(xy, r=15.0)
            # scene_goal = scene_goal[mask]

            if args.normalize_scene:
                xy, rotation, center, scene_goal = center_scene(xy, obs_length, goals=scene_goal)
            
            xy = torch.Tensor(xy)  #.to(self.device)
            scene_goal = torch.Tensor(scene_goal) #.to(device)
            batch_split = torch.Tensor(batch_split).long()

            # _, output_scenes = self.model(xy[start_length:obs_length], scene_goal, batch_split, xy[obs_length:-1].clone())
            _, output_scenes, _ = self.model(xy[start_length:obs_length], scene_goal, batch_split, n_predict=n_predict)

            multimodal_outputs = {}
            for mode in range(self.model.num_modes):
                output_scenes_m = output_scenes[mode].numpy()
                if args.normalize_scene:
                    output_scenes_m = inverse_scene(output_scenes_m, rotation, center)
                output_primary = output_scenes_m[-n_predict:, 0]
                output_neighs = output_scenes_m[-n_predict:, 1:]
                if mode == 0:
                    ## Dictionary of predictions. Each key corresponds to one mode
                    multimodal_outputs[mode] = [output_primary, output_neighs]
                else:
                    multimodal_outputs[mode] = [output_primary, []]

        self.model.train()
        ## Return Dictionary of predictions. Each key corresponds to one mode
        return multimodal_outputs




class VAEEncoder(torch.nn.Module):
    """C-VAE Encoder

    """
    def __init__(self, input_dims, output_dim):
        super(VAEEncoder, self).__init__()
        # (sqrt(2*hidden_dim), sqrt(2*hidden_dim))
        self.input_dims = input_dims
        self.latent_dim = output_dim // 2
        self.conv = torch.nn.Sequential(collections.OrderedDict([
          ('conv1', torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)),
          ('conv2', torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)),
          ('conv3', torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)),
        ]))
        self.fc_mu = torch.nn.Linear(in_features=128, out_features=self.latent_dim)
        self.fc_var = torch.nn.Linear(in_features=128, out_features=self.latent_dim)

    def forward(self, inputs):
        inputs = torch.stack(inputs)
        inputs = torch.reshape(inputs, shape=(-1, 1, self.input_dims[0], self.input_dims[1]))
        inputs = self.conv(inputs)
        inputs = torch.reshape(inputs, shape=(-1, 128))
        z_mu = self.fc_mu(inputs)
        z_log_var = self.fc_var(inputs)
        
        # Logarithm of the variance
        # this allows for smoother representations for the latent space.
        return z_mu, z_log_var


class VAEDecoder(torch.nn.Module):
    """C-VAE Decoder

    """
    def __init__(self, input_dim, output_dim):
        super(VAEDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = torch.nn.Sequential(collections.OrderedDict([
          ('conv1', torch.nn.ConvTranspose2d(in_channels=self.input_dim, out_channels=64, kernel_size=5)),
          ('conv2', torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2)),
          ('conv3', torch.nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=5))
        ]))
        self.fc = torch.nn.Linear(in_features=256, out_features=self.output_dim)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, inputs):
        inputs = torch.reshape(inputs, shape=(-1, self.input_dim, 1, 1))
        inputs = self.conv(inputs)
        inputs = torch.reshape(inputs, shape=(-1, 256))
        return self.softmax(self.relu(self.fc(inputs)))