import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt
import pickle

from pytorch3dunet.datasets.utils import get_train_loaders
from pytorch3dunet.unet3d.losses import get_loss_criterion
from pytorch3dunet.unet3d.metrics import get_evaluation_metric
from pytorch3dunet.unet3d.model import get_model, UNet2D
from pytorch3dunet.unet3d.utils import get_logger, get_tensorboard_formatter, create_optimizer, \
    create_lr_scheduler, get_number_of_learnable_parameters
from . import utils

logger = get_logger('UNetTrainer')


def create_trainer(config):
    # Create the model
    model = get_model(config['model'])

    if torch.cuda.device_count() > 1 and not config['device'] == 'cpu':
        model = nn.DataParallel(model)
        logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction')
    if torch.cuda.is_available() and not config['device'] == 'cpu':
        model = model.cuda()

    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # Create loss criterion
    loss_criterion = get_loss_criterion(config)
    # Create evaluation metric
    eval_criterion = get_evaluation_metric(config)

    # Create data loaders
    loaders = get_train_loaders(config)

    # Create the optimizer
    optimizer = create_optimizer(config['optimizer'], model)

    # Create learning rate adjustment strategy
    lr_scheduler = create_lr_scheduler(config.get('lr_scheduler', None), optimizer)

    trainer_config = config['trainer']
    # Create tensorboard formatter
    tensorboard_formatter = get_tensorboard_formatter(trainer_config.pop('tensorboard_formatter', None))
    # Create trainer
    resume = trainer_config.pop('resume', None)
    pre_trained = trainer_config.pop('pre_trained', None)

    return UNetTrainer(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, loss_criterion=loss_criterion,
                       eval_criterion=eval_criterion, loaders=loaders, tensorboard_formatter=tensorboard_formatter,
                       resume=resume, pre_trained=pre_trained, **trainer_config)


class UNetTrainer:
    """UNet trainer.

    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metric (such as Dice, IoU, AP or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        # max_num_iterations (int): maximum number of iterations
        # validate_after_iters (int): validate after that many iterations
        # log_after_iters (int): number of iterations before logging to tensorboard
        # validate_iters (int): number of validation iterations, if None validate
        #     on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
        tensorboard_formatter (callable): converts a given batch of input/output/target image to a series of images
            that can be displayed in tensorboard
        skip_train_validation (bool): if True eval_criterion is not evaluated on the training set (used mostly when
            evaluation is expensive)
    """

    def __init__(self, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders, checkpoint_dir,
                 max_num_epochs, num_iterations=1, num_epoch=0, checkpoint_after_epochs=5,
                 eval_score_higher_is_better=True, tensorboard_formatter=None,
                 skip_train_validation=False, resume=None, pre_trained=None, **kwargs):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.checkpoint_after_epochs = checkpoint_after_epochs
        self.eval_score_higher_is_better = eval_score_higher_is_better
        
        # TODO/ADDON: torch tensor where columns are 1: loss, 2: eval score
        self.train_stats = torch.zeros((self.max_num_epochs, 2))
        self.val_stats = torch.zeros((self.max_num_epochs, 2))

        logger.info(model)
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        # initialize the best_eval_score
        if eval_score_higher_is_better:
            self.best_eval_score = float('-inf')
        else:
            self.best_eval_score = float('+inf')

        self.writer = SummaryWriter(
            log_dir=os.path.join(
                checkpoint_dir, 'logs', 
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                )
            )

        assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        self.tensorboard_formatter = tensorboard_formatter

        self.num_iterations = num_iterations
        self.num_epochs = num_epoch
        self.skip_train_validation = skip_train_validation

        if resume is not None:
            logger.info(f"Loading checkpoint '{resume}'...")
            state = utils.load_checkpoint(resume, self.model, self.optimizer)
            logger.info(
                f"Checkpoint loaded from '{resume}'. Epoch: {state['num_epochs']}.  Iteration: {state['num_iterations']}. "
                f"Best val score: {state['best_eval_score']}."
            )
            self.best_eval_score = state['best_eval_score']
            self.num_iterations = state['num_iterations']
            self.num_epochs = state['num_epochs']
            # extending loss and eval scores to fit max_num_epochs
            empty_stats = torch.zeros((self.max_num_epochs - self.num_epochs, 2))
            self.train_stats = torch.cat([state['train_stats'], empty_stats], dim=0)
            self.val_stats = torch.cat([state['val_stats'], empty_stats], dim=0)
            self.checkpoint_dir = os.path.split(resume)[0]
        elif pre_trained is not None:
            logger.info(f"Logging pre-trained model from '{pre_trained}'...")
            utils.load_checkpoint(pre_trained, self.model, None)
            if 'checkpoint_dir' not in kwargs:
                self.checkpoint_dir = os.path.split(pre_trained)[0]

    def fit(self):
        for epoch in range(self.num_epochs, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train()

            # set the model in eval mode
            self.model.eval()
            # evaluate on validation set
            eval_score = self.validate()
            # set the model back to training mode
            self.model.train()

            # adjust learning rate if necessary
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(eval_score)
            elif self.scheduler is not None:
                self.scheduler.step()

            # log current learning rate in tensorboard
            self._log_lr()
            # remember best validation metric
            is_best = self._is_best_eval_score(eval_score)

            # save checkpoint
            if is_best or (epoch % self.checkpoint_after_epochs == self.checkpoint_after_epochs - 1):
                self._save_checkpoint(is_best)  
                self._save_stats_graph()          

            if should_terminate:
                logger.info('Stopping criterion is satisfied. Finishing training')
                return

            self.num_epochs += 1
        logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")

    def train(self):
        """Trains the model for 1 epoch.

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        train_losses = utils.RunningAverage()
        train_eval_scores = utils.RunningAverage()

        # sets the model in training mode
        self.model.train()

        for t in self.loaders['train']:
            logger.info(f'Training iteration [{self.num_iterations}]. '
                        f'Epoch [{self.num_epochs}/{self.max_num_epochs - 1}]')

            input, target, weight = self._split_training_batch(t)

            output, loss = self._forward_pass(input, target, weight)

            train_losses.update(loss.item(), self._batch_size(input))

            # compute gradients and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # compute eval criterion
            if not self.skip_train_validation:
                # apply final activation before calculating eval score
                if isinstance(self.model, nn.DataParallel):
                    final_activation = self.model.module.final_activation
                else:
                    final_activation = self.model.final_activation

                if final_activation is not None:
                    act_output = final_activation(output)
                else:
                    act_output = output
                eval_score = self.eval_criterion(act_output, target)
                train_eval_scores.update(eval_score.item(), self._batch_size(input))

            if self.should_stop():
                return True

            self.num_iterations += 1

        # log stats, params and images
        logger.info(
            f'Training stats. Loss: {train_losses.avg}. Evaluation score: {train_eval_scores.avg}')
        self._log_stats('train', train_losses.avg, train_eval_scores.avg)
        # self._log_params()
        # self._log_images(input, target, output, 'train_')

        # TODO/ADDON: updating training statistics
        self.train_stats[self.num_epochs, :] = torch.tensor((train_losses.avg, train_eval_scores.avg))

        return False

    def should_stop(self):
        """
        Training will terminate if the learning rate drops below some predefined threshold (1e-6 in our case)
        """
        min_lr = 1e-6
        lr = self.optimizer.param_groups[0]['lr']
        if lr < min_lr:
            logger.info(f'Learning rate below the minimum {min_lr}.')
            return True

        return False

    def validate(self):
        logger.info('Validating...')

        val_losses = utils.RunningAverage()
        val_scores = utils.RunningAverage()

        with torch.no_grad():
            for i, t in enumerate(self.loaders['val']):
                logger.info(f'Validation iteration {i}')

                input, target, weight = self._split_training_batch(t)

                output, loss = self._forward_pass(input, target, weight)
                val_losses.update(loss.item(), self._batch_size(input))

                if i % 100 == 0:
                    self._log_images(input, target, output, 'val_')

                eval_score = self.eval_criterion(output, target)
                val_scores.update(eval_score.item(), self._batch_size(input))

            self._log_stats('val', val_losses.avg, val_scores.avg)
            logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
            
            # TODO/ADDON: updating validation statistics
            self.val_stats[self.num_epochs, :] = torch.tensor((val_losses.avg, val_scores.avg))

            return val_scores.avg

    def _split_training_batch(self, t):
        def _move_to_gpu(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_gpu(x) for x in input])
            else:
                if torch.cuda.is_available():
                    input = input.cuda(non_blocking=True)
                return input

        t = _move_to_gpu(t)
        weight = None
        if len(t) == 2:
            input, target = t
        else:
            input, target, weight = t
        return input, target, weight

    def _forward_pass(self, input, target, weight=None):
        if isinstance(self.model, UNet2D):
            # remove the singleton z-dimension from the input
            input = torch.squeeze(input, dim=-3)
            # forward pass
            output = self.model(input)
            # add the singleton z-dimension to the output
            output = torch.unsqueeze(output, dim=-3)
        else:
            # forward pass
            output = self.model(input)

        # compute the loss
        if weight is None:
            loss = self.loss_criterion(output, target)
        else:
            loss = self.loss_criterion(output, target, weight)

        return output, loss

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        last_file_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pytorch')
        logger.info(f"Saving checkpoint to '{last_file_path}'")

        utils.save_checkpoint({
            'num_epochs': self.num_epochs + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': state_dict,
            'best_eval_score': self.best_eval_score,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_stats': self.train_stats, 
            'val_stats': self.val_stats
        }, is_best, checkpoint_dir=self.checkpoint_dir)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase, loss_avg, eval_score_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction, prefix=''):

        if isinstance(self.model, nn.DataParallel):
            net = self.model.module
        else:
            net = self.model

        if net.final_activation is not None:
            prediction = net.final_activation(prediction)

        inputs_map = {
            'inputs': input,
            'targets': target,
            'predictions': prediction
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(prefix + tag, image, self.num_iterations)

    # TODO/ADDON: creating and saving figures for loss, eval score, learning rate, and time per epoch
    def _plot_stats(self, stat):
        # subsetting train and val stats tensors to just losses or eval score
        col_idx = None
        if stat.lower() == 'loss':
            col_idx = 0
        elif stat.lower() == 'score':
            col_idx = 1
        else:
            raise Exception("stat argument can only be either 'loss' or 'score.'")
        
        stat = stat.capitalize()
        epochs = torch.arange(self.max_num_epochs)

        # plotting figure
        fig, ax = plt.subplots()
        ax.plot(epochs, self.train_stats[:, col_idx], color='blue', label=f'Train {stat}')
        ax.plot(epochs, self.val_stats[:, col_idx], color='red', label=f'Val {stat}')
        ax.set(xlabel='Epochs', ylabel=stat)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        # saving figure and pickling it
        file_name = os.path.join(self.checkpoint_dir, f'{stat}_Plot')
        fig.savefig(f'{file_name}.png')
        with open(f'{file_name}.pkl', 'wb') as f:
            pickle.dump(fig, f)

    def _save_stats_graph(self):
        self._plot_stats('loss')
        self._plot_stats('score')

    def _save_lr_graph():    
        pass 

    def _save_time_per_epoch_graph():
        pass

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)
