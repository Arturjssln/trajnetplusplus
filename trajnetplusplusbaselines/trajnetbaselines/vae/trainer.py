"""Command line tool to train an VAE model."""

import argparse
import logging
import socket
import sys
import time
import random
import os
import pickle
import torch
import numpy as np

import trajnetplusplustools

from .vae import VAE, VAEPredictor
from .. import augmentation
from .loss import PredictionLoss, L2Loss, KLDLoss, ReconstructionLoss, VarietyLoss, L1Loss
from .utils import drop_distant
from .pooling.non_gridbased_pooling import NN_Pooling
from .pooling.fast_non_gridbased_pooling import NN_Pooling_fast

from .. import __version__ as VERSION

from .utils import center_scene, random_rotation



class Trainer(object):
    def __init__(self, model=None, criterion='L2', multimodal_criterion='recon', optimizer=None, 
                lr_scheduler=None, device=None, batch_size=32, obs_length=9, pred_length=12, 
                augment=False, normalize_scene=False, save_every=1, start_length=0, 
                obs_dropout=False, alpha_kld=1, num_modes=1, desire_approach=False,
                fast_parallel=False):
        self.model = model if model is not None else VAE()
        if criterion == 'L2':
            self.criterion = L2Loss()
            self.loss_multiplier = 100
        elif criterion == 'L1':
            self.criterion = L1Loss()
            self.loss_multiplier = 1
        else:
            self.criterion = PredictionLoss()
            self.loss_multiplier = 1
        
        self.kld_loss = KLDLoss()
        self.alpha_kld = alpha_kld
        self.num_modes = num_modes
        self.desire_approach = desire_approach

        self.optimizer = optimizer if optimizer is not None else torch.optim.SGD(
            self.model.parameters(), lr=3e-4, momentum=0.9)
        self.lr_scheduler = (lr_scheduler
                             if lr_scheduler is not None
                             else torch.optim.lr_scheduler.StepLR(self.optimizer, 15))

        self.device = device if device is not None else torch.device('cpu')
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.log = logging.getLogger(self.__class__.__name__)
        self.save_every = save_every

        self.batch_size = batch_size
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.seq_length = self.obs_length+self.pred_length

        if multimodal_criterion == 'variety':
            self.multimodal_criterion = VarietyLoss(self.criterion, self.pred_length, self.loss_multiplier)
        else:
            self.multimodal_criterion = ReconstructionLoss(self.criterion, self.pred_length, self.loss_multiplier, self.batch_size, self.num_modes)
        

        self.augment = augment
        self.normalize_scene = normalize_scene

        self.start_length = start_length
        self.obs_dropout = obs_dropout

        self.fast_parallel = fast_parallel

    def loop(self, train_scenes, val_scenes, train_goals, val_goals, out, epochs=35, start_epoch=0):
        epoch = 0
        for epoch in range(start_epoch, start_epoch + epochs):
            if epoch % self.save_every == 0:
                state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'scheduler': self.lr_scheduler.state_dict()}
                VAEPredictor(self.model).save(state, out + f'.epoch{epoch}')
            self.train(train_scenes, train_goals, epoch)
            self.val(val_scenes, val_goals, epoch)


        state = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'scheduler': self.lr_scheduler.state_dict()}
        VAEPredictor(self.model).save(state, out + f'.epoch{epoch+1}')
        VAEPredictor(self.model).save(state, out)

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def train(self, scenes, goals, epoch):
        start_time = time.time()

        print('epoch', epoch)

        random.shuffle(scenes)
        epoch_loss = 0.0
        self.model.train()
        self.optimizer.zero_grad()

        ## Initialize batch of scenes
        batch_scene = []
        batch_scene_goal = []
        batch_split = [0]

        for scene_i, (filename, scene_id, paths) in enumerate(scenes):
            scene_start = time.time()

            ## make new scene
            scene = trajnetplusplustools.Reader.paths_to_xy(paths)

            ## get goals
            if goals is not None:
                scene_goal = np.array(goals[filename][scene_id])
            else:
                scene_goal = np.array([[0, 0] for path in paths])

            if not self.fast_parallel:
                ## Drop Distant
                scene, mask = drop_distant(scene)
                scene_goal = scene_goal[mask]

            ## Process scene
            if self.normalize_scene:
                scene, _, _, scene_goal = center_scene(scene, self.obs_length, goals=scene_goal)
            if self.augment:
                scene, scene_goal = random_rotation(scene, goals=scene_goal)
                # scene = augmentation.add_noise(scene, thresh=0.01)

            ## Augment scene to batch of scenes
            batch_scene.append(scene)
            batch_split.append(int(scene.shape[1]))
            batch_scene_goal.append(scene_goal)

            if ((scene_i + 1) % self.batch_size == 0) or ((scene_i + 1) == len(scenes)):
                ## Construct Batch
                batch_scene = np.concatenate(batch_scene, axis=1)
                batch_scene_goal = np.concatenate(batch_scene_goal, axis=0)
                batch_split = np.cumsum(batch_split)
                batch_scene = torch.Tensor(batch_scene).to(self.device)
                batch_scene_goal = torch.Tensor(batch_scene_goal).to(self.device)
                batch_split = torch.Tensor(batch_split).to(self.device).long()

                preprocess_time = time.time() - scene_start

                ## Train Batch
                loss = self.train_batch(batch_scene, batch_scene_goal, batch_split)
                epoch_loss += loss
                total_time = time.time() - scene_start

                ## Reset Batch
                batch_scene = []
                batch_scene_goal = []
                batch_split = [0]

                if (scene_i + 1) % (10*self.batch_size) == 0:
                    self.log.info({
                        'type': 'train',
                        'epoch': epoch, 'batch': scene_i, 'n_batches': len(scenes),
                        'time': round(total_time, 3),
                        'data_time': round(preprocess_time, 3),
                        'lr': self.get_lr(),
                        'loss': round(loss, 3),
                    })

        self.lr_scheduler.step()

        self.log.info({
            'type': 'train-epoch',
            'epoch': epoch + 1,
            'loss': round(epoch_loss / (len(scenes)), 5),
            'time': round(time.time() - start_time, 1),
        })

    def val(self, scenes, goals, epoch):
        eval_start = time.time()

        val_loss = 0.0
        test_loss = 0.0
        self.model.train()

        ## Initialize batch of scenes
        batch_scene = []
        batch_scene_goal = []
        batch_split = [0]

        for scene_i, (filename, scene_id, paths) in enumerate(scenes):
            ## make new scene
            scene = trajnetplusplustools.Reader.paths_to_xy(paths)

            ## get goals
            if goals is not None:
                scene_goal = np.array(goals[filename][scene_id])
            else:
                scene_goal = np.array([[0, 0] for path in paths])

            if not self.fast_parallel:
                ## Drop Distant
                scene, mask = drop_distant(scene)
                scene_goal = scene_goal[mask]

            ## Process scene
            if self.normalize_scene:
                scene, _, _, scene_goal = center_scene(scene, self.obs_length, goals=scene_goal)

            ## Augment scene to batch of scenes
            batch_scene.append(scene)
            batch_split.append(int(scene.shape[1]))
            batch_scene_goal.append(scene_goal)

            if ((scene_i + 1) % self.batch_size == 0) or ((scene_i + 1) == len(scenes)):
                ## Construct Batch
                batch_scene = np.concatenate(batch_scene, axis=1)
                batch_scene_goal = np.concatenate(batch_scene_goal, axis=0)
                batch_split = np.cumsum(batch_split)
                
                batch_scene = torch.Tensor(batch_scene).to(self.device)
                batch_scene_goal = torch.Tensor(batch_scene_goal).to(self.device)
                batch_split = torch.Tensor(batch_split).to(self.device).long()
                
                loss_val_batch, loss_test_batch = self.val_batch(batch_scene, batch_scene_goal, batch_split)
                #loss_val_batch = self.val_batch(batch_scene, batch_scene_goal, batch_split)
                val_loss += loss_val_batch
                test_loss += loss_test_batch

                ## Reset Batch
                batch_scene = []
                batch_scene_goal = []
                batch_split = [0]

        eval_time = time.time() - eval_start

        self.log.info({
            'type': 'val-epoch',
            'epoch': epoch + 1,
            'loss': round(val_loss / (len(scenes)), 3),
            'test_loss': round(test_loss / len(scenes), 3), 
            'time': round(eval_time, 1),
        })

    def train_batch(self, batch_scene, batch_scene_goal, batch_split):
        """Training of B batches in parallel, B : batch_size

        Parameters
        ----------
        batch_scene : Tensor [seq_length, num_tracks, 2]
            Tensor of batch of scenes.
        batch_scene_goal : Tensor [num_tracks, 2]
            Tensor of goals of each track in batch
        batch_split : Tensor [batch_size + 1]
            Tensor defining the split of the batch.
            Required to identify the tracks of to the same scene


        Returns
        -------
        loss : scalar
            Training loss of the batch
        """

        ## If observation dropout active
        if self.obs_dropout:
            self.start_length = random.randint(0, self.obs_length - 2)

        observed = batch_scene[self.start_length:self.obs_length].clone()
        prediction_truth = batch_scene[self.obs_length:self.seq_length-1].clone()
        targets = batch_scene[self.obs_length:self.seq_length] - batch_scene[self.obs_length-1:self.seq_length-1]

        rel_outputs, _, z_distr_xy, z_distr_x = self.model(observed, batch_scene_goal, batch_split, prediction_truth)

        ## Loss wrt primary tracks of each scene only

        # Multimodal loss
        multimodal_loss = self.multimodal_criterion(rel_outputs, targets, batch_split)

        # TODO: remove
        #reconstr_loss = 0
        #for rel_outputs_mode in rel_outputs:
        #    reconstr_loss += self.criterion(rel_outputs_mode[-self.pred_length:], targets, batch_split) * self.batch_size * self.loss_multiplier / self.num_modes
        
        # KLD loss
        kld_loss = self.kld_loss(inputs=z_distr_xy, targets=z_distr_x) * self.batch_size
        
        ## Total loss is the sum of the multimodal loss and the kld loss
        loss = multimodal_loss + self.alpha_kld * kld_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def val_batch(self, batch_scene, batch_scene_goal, batch_split):
        """Validation of B batches in parallel, B : batch_size

        Parameters
        ----------
        batch_scene : Tensor [seq_length, num_tracks, 2]
            Tensor of batch of scenes.
        batch_scene_goal : Tensor [num_tracks, 2]
            Tensor of goals of each track in batch
        batch_split : Tensor [batch_size + 1]
            Tensor defining the split of the batch.
            Required to identify the tracks of to the same scene

        Returns
        -------
        loss : scalar
            Validation loss of the batch when groundtruth of neighbours
            is provided
        loss_test : scalar
            Validation loss of the batch when groundtruth of neighbours
            is not provided (evaluation scenario)
        """

        if self.obs_dropout:
            self.start_length = 0
        observed = batch_scene[self.start_length:self.obs_length]
        prediction_truth = batch_scene[self.obs_length:self.seq_length-1].clone()  ## CLONE
        targets = batch_scene[self.obs_length:self.seq_length] - batch_scene[self.obs_length-1:self.seq_length-1]
        observed_test = observed.clone()
        with torch.no_grad():
            ## groundtruth of neighbours provided (Better validation curve to monitor model)
            rel_outputs, _, z_distr_xy, z_distr_x = self.model(observed, batch_scene_goal, batch_split, prediction_truth)
            
            # Multimodal loss
            multimodal_loss = self.multimodal_criterion(rel_outputs, targets, batch_split)

            # TODO: remove
            #reconstr_loss = 0
            #for rel_outputs_mode in rel_outputs:
            #    reconstr_loss += self.criterion(rel_outputs_mode[-self.pred_length:], targets, batch_split) * self.batch_size * self.loss_multiplier / self.num_modes
            kld_loss = self.kld_loss(inputs=z_distr_xy, targets=z_distr_x) * self.batch_size * self.loss_multiplier
            loss = multimodal_loss + self.alpha_kld * kld_loss
            
            ## groundtruth of neighbours not provided 
            self.model.eval()

            #loss_test = 0
            rel_outputs_test, _, _, _ = self.model(observed_test, batch_scene_goal, batch_split, n_predict=self.pred_length)
            # Multimodal loss
            loss_test = self.multimodal_criterion(rel_outputs_test, targets, batch_split)
            # TODO: remvoe
            #for rel_outputs_mode in rel_outputs_test:
            #    loss_test += self.criterion(rel_outputs_mode[-self.pred_length:], targets, batch_split) * self.batch_size * self.loss_multiplier / self.num_modes
            self.model.train()
        return loss.item(), loss_test.item()

def prepare_data(path, subset='/train/', sample=1.0, goals=True, goal_files='goal_files'):
    """ Prepares the train/val scenes and corresponding goals
    
    Parameters
    ----------
    subset: String ['/train/', '/val/']
        Determines the subset of data to be processed
    sample: Float (0.0, 1.0]
        Determines the ratio of data to be sampled
    goals: Bool
        If true, the goals of each track are extracted
        The corresponding goal file must be present in the goal_files folder
        The name of the goal file must be the same as the name of the training file
    goal_files : String
        Path to goal files data
        Default: 'goal_files'
    Returns
    -------
    all_scenes: List
        List of all processed scenes
    all_goals: Dictionary
        Dictionary of goals corresponding to each dataset file.
        None if 'goals' argument is False.
    """

    ## read goal files
    all_goals = {}
    all_scenes = []

    ## List file names
    files = [f.split('.')[-2] for f in os.listdir(path + subset) if f.endswith('.ndjson')]
    ## Iterate over file names
    for file in files:
        reader = trajnetplusplustools.Reader(path + subset + file + '.ndjson', scene_type='paths')
        ## Necessary modification of train scene to add filename
        scene = [(file, s_id, s) for s_id, s in reader.scenes(sample=sample)]
        if goals:
            goal_dict = pickle.load(open(goal_files + subset + file +'.pkl', "rb"))
            ## Get goals corresponding to train scene
            all_goals[file] = {s_id: [goal_dict[path[0].pedestrian] for path in s] for _, s_id, s in scene}
        all_scenes += scene

    if goals:
        return all_scenes, all_goals
    return all_scenes, None

def main(epochs=50):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=epochs, type=int,
                        help='number of epochs')
    parser.add_argument('--step_size', default=15, type=int,
                        help='step_size of lr scheduler')
    parser.add_argument('--save_every', default=1, type=int,
                        help='frequency of saving model (in terms of epochs)')
    parser.add_argument('--obs_length', default=9, type=int,
                        help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,
                        help='prediction length')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--type', default='vanilla',
                        choices=('vanilla', 'occupancy', 'directional', 'social', 'hiddenstatemlp', 's_att_fast',
                                 'directionalmlp', 'nn', 'attentionmlp', 'nn_lstm', 'traj_pool', 'nmmp'),
                        help='type of interaction encoder')
    parser.add_argument('--norm_pool', action='store_true',
                        help='normalize the scene along direction of movement')
    parser.add_argument('--front', action='store_true',
                        help='Front pooling (only consider pedestrian in front along direction of movement)')
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--augment', action='store_true',
                        help='augment scenes (rotation augmentation)')
    parser.add_argument('--normalize_scene', action='store_true',
                        help='rotate scene so primary pedestrian moves northwards at end of oservation')
    parser.add_argument('--path', default='trajdata',
                        help='glob expression for data files')
    parser.add_argument('--goal_path', default=None,
                        help='glob expression for goal files')
    parser.add_argument('--loss', default='L2', choices=('L2', 'gauss', 'L1'),
                        help='loss objective to train the model')
    parser.add_argument('--multi_loss', default='recon', choices=('recon', 'variety'),
                        help='multimodal loss objective to train the model')
    parser.add_argument('--goals', action='store_true',
                        help='flag to use goals')
    parser.add_argument('--num_modes', default=1, type=int,
                        help='Number of modes for reconstruction loss') 
    parser.add_argument('--goal_files', default='goal_files',
                        help='Path for goal files') 
    parser.add_argument('--desire', action='store_true',
                        help='Use Desire approach (Trajectron by default)') 
    parser.add_argument('--noise', default='default', choices=('default', 'product', 'concat', 'additive'),
                        help='Inclusion of noise approach (Default for DESIRE: h*z, for Trajectron: [h,z])') 
    parser.add_argument('--dis_value', default=None, type=float,
                        help='Value by which the first element of the primary in the latent space is replaced') 
    parser.add_argument('--fast_parallel', action='store_true',
                        help='Use Fast parallel pooling') 
    pretrain = parser.add_argument_group('pretraining')
    pretrain.add_argument('--load-state', default=None,
                          help='load a pickled model state dictionary before training')
    pretrain.add_argument('--load-full-state', default=None,
                          help='load a pickled full state dictionary before training')
    pretrain.add_argument('--nonstrict-load-state', default=None,
                          help='load a pickled state dictionary before training')

    ##Pretrain Pooling AE
    pretrain.add_argument('--load_pretrained_pool_path', default=None,
                          help='load a pickled model state dictionary of pool AE before training')
    pretrain.add_argument('--pretrained_pool_arch', default='onelayer',
                          help='architecture of pool representation')
    pretrain.add_argument('--downscale', type=int, default=4,
                          help='downscale factor of pooling grid')
    pretrain.add_argument('--finetune', type=int, default=0,
                          help='finetune factor of pretrained model')

    hyperparameters = parser.add_argument_group('hyperparameters')
    hyperparameters.add_argument('--hidden-dim', type=int, default=128,
                                 help='LSTM hidden dimension')
    hyperparameters.add_argument('--coordinate-embedding-dim', type=int, default=64,
                                 help='coordinate embedding dimension')
    hyperparameters.add_argument('--cell_side', type=float, default=0.6,
                                 help='cell size of real world')
    hyperparameters.add_argument('--n', type=int, default=16,
                                 help='number of cells per side')
    hyperparameters.add_argument('--layer_dims', type=int, nargs='*', default=[512],
                                 help='interaction module layer dims (for gridbased pooling)')
    hyperparameters.add_argument('--pool_dim', type=int, default=256,
                                 help='output dimension of pooling/interaction vector')
    hyperparameters.add_argument('--embedding_arch', default='two_layer',
                                 help='interaction encoding arch for gridbased pooling')
    hyperparameters.add_argument('--goal_dim', type=int, default=64,
                                 help='goal dimension')
    hyperparameters.add_argument('--spatial_dim', type=int, default=32,
                                 help='attentionmlp spatial dimension')
    hyperparameters.add_argument('--vel_dim', type=int, default=32,
                                 help='attentionmlp vel dimension')
    hyperparameters.add_argument('--no_vel', action='store_true',
                                 help='flag to not consider velocity in nn')
    hyperparameters.add_argument('--pool_constant', default=0, type=int,
                                 help='background value of gridbased pooling')
    hyperparameters.add_argument('--sample', default=1.0, type=float,
                                 help='sample ratio of train/val scenes')
    hyperparameters.add_argument('--norm', default=0, type=int,
                                 help='normalization scheme for grid-based')
    hyperparameters.add_argument('--neigh', default=4, type=int,
                                 help='number of neighbours to consider in DirectConcat')
    hyperparameters.add_argument('--mp_iters', default=5, type=int,
                                 help='message passing iters in NMMP')
    hyperparameters.add_argument('--obs_dropout', action='store_true',
                                 help='obs length dropout (regularization)')
    hyperparameters.add_argument('--start_length', default=0, type=int,
                                 help='start length during obs dropout')
    hyperparameters.add_argument('--alpha_kld', default=1, type=float,
                                 help='multiplier coefficient for kld loss')                      
    args = parser.parse_args()

    ## Fixed set of scenes if sampling
    if args.sample < 1.0:
        torch.manual_seed("080819")
        random.seed(1)

    ## Define location to save trained model
    if not os.path.exists(f'OUTPUT_BLOCK/{args.path}'):
        os.makedirs(f'OUTPUT_BLOCK/{args.path}')
    goals = '_goals' if args.goals else ''
    fast_paralell = '_parallel' if args.fast_parallel else ''
    args.output = f'OUTPUT_BLOCK/{args.path}/vae{fast_paralell}{goals}_{args.type}_{args.output}_{args.num_modes}.pkl'

    # configure logging
    from pythonjsonlogger import jsonlogger
    if args.load_full_state:
        file_handler = logging.FileHandler(args.output + '.log', mode='a')
    else:
        file_handler = logging.FileHandler(args.output + '.log', mode='w')
    file_handler.setFormatter(jsonlogger.JsonFormatter('(message) (levelname) (name) (asctime)'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler])
    logging.info({
        'type': 'process',
        'argv': sys.argv,
        'args': vars(args),
        'version': VERSION,
        'hostname': socket.gethostname(),
    })

    # refactor args for --load-state
    # loading a previously saved model
    args.load_state_strict = True
    if args.nonstrict_load_state:
        args.load_state = args.nonstrict_load_state
        args.load_state_strict = False
    if args.load_full_state:
        args.load_state = args.load_full_state

    # add args.device
    args.device = torch.device('cpu')
    # if not args.disable_cuda and torch.cuda.is_available():
    #     args.device = torch.device('cuda')

    args.path = 'DATA_BLOCK/' + args.path
    ## Prepare data
    train_scenes, train_goals = prepare_data(args.path, subset='/train/', sample=args.sample, goals=args.goals, goal_files=args.goal_files)
    val_scenes, val_goals = prepare_data(args.path, subset='/val/', sample=args.sample, goals=args.goals, goal_files=args.goal_files)

    ## pretrained pool model (if any)
    pretrained_pool = None

    # create interaction/pooling modules
    pool = None
    if args.type == 'nn':
        if args.fast_parallel:
            pool = NN_Pooling_fast(n=args.neigh, out_dim=args.pool_dim)
        else:
            pool = NN_Pooling(n=args.neigh, out_dim=args.pool_dim, no_vel=args.no_vel)
  
    # create forecasting model
    model = VAE(pool=pool,
                 embedding_dim=args.coordinate_embedding_dim,
                 hidden_dim=args.hidden_dim,
                 goal_flag=args.goals,
                 goal_dim=args.goal_dim,
                 num_modes=args.num_modes,
                 desire_approach=args.desire,
                 noise_approach=args.noise,
                 disentangling_value=args.dis_value,
                 fast_parallel=args.fast_parallel)

    # optimizer and schedular
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = None
    if args.step_size is not None:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size)
    start_epoch = 0

    # train
    if args.load_state:
        # load pretrained model.
        # useful for tranfer learning
        print("Loading Model Dict")
        with open(args.load_state, 'rb') as f:
            checkpoint = torch.load(f)
        pretrained_state_dict = checkpoint['state_dict']
        model.load_state_dict(pretrained_state_dict, strict=args.load_state_strict)

        if args.load_full_state:
        # load optimizers from last training
        # useful to continue model training
            print("Loading Optimizer Dict")
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) # , weight_decay=1e-4
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15)
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']

    #trainer
    training_start_time = time.time()
    trainer = Trainer(model, optimizer=optimizer, lr_scheduler=lr_scheduler, device=args.device,
                      criterion=args.loss, multimodal_criterion=args.multi_loss, batch_size=args.batch_size, obs_length=args.obs_length,
                      pred_length=args.pred_length, augment=args.augment, normalize_scene=args.normalize_scene,
                      save_every=args.save_every, start_length=args.start_length, obs_dropout=args.obs_dropout,
                      alpha_kld=args.alpha_kld, num_modes=args.num_modes, desire_approach=args.desire,
                      fast_parallel=args.fast_parallel)
    trainer.loop(train_scenes, val_scenes, train_goals, val_goals, args.output, epochs=args.epochs, start_epoch=start_epoch)

    print({
            'type': 'trainer',
            'fast_parallel': args.fast_parallel,
            'total_training_time_min': round((time.time() - training_start_time)/60, 1),
        })

if __name__ == '__main__':
    main()
