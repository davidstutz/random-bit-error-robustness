import torch
import numpy
import common.torch
import common.summary
import common.numpy
import common.quantization
from common.log import log
import attacks
from imgaug import augmenters as iaa
from .training_interface import TrainingInterface


class NormalTraining(TrainingInterface):
    """
    Normal training.
    """

    def __init__(self, model, trainset, testset, optimizer, scheduler, augmentation=None, loss=common.torch.classification_loss, writer=common.summary.SummaryWriter(), cuda=False):
        """
        Constructor.

        :param model: model
        :type model: torch.nn.Module
        :param trainset: training set
        :type trainset: torch.utils.data.DataLoader
        :param testset: test set
        :type testset: torch.utils.data.DataLoader
        :param optimizer: optimizer
        :type optimizer: torch.optim.Optimizer
        :param scheduler: scheduler
        :type scheduler: torch.optim.LRScheduler
        :param augmentation: augmentation
        :type augmentation: imgaug.augmenters.Sequential
        :param writer: summary writer
        :type writer: torch.utils.tensorboard.SummaryWriter or TensorboardX equivalent
        :param cuda: run on CUDA device
        :type cuda: bool
        """

        assert loss is not None
        assert callable(loss)
        assert isinstance(model, torch.nn.Module)
        assert len(trainset) > 0
        assert len(testset) > 0
        assert isinstance(trainset, torch.utils.data.DataLoader)
        assert isinstance(testset, torch.utils.data.DataLoader)
        assert isinstance(testset.sampler, torch.utils.data.SequentialSampler)
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler)
        assert (cuda and common.torch.is_cuda(model)) or (not cuda and not common.torch.is_cuda(model))

        super(NormalTraining, self).__init__(writer)

        self.model = model
        """ (torch.nn.Module) Model. """

        parameters = list(self.model.parameters())
        self.layers = [i for i in range(len(parameters)) if parameters[i].requires_grad is True]
        """ ([int]) Layers for projection. """

        self.trainset = trainset
        """ (torch.utils.data.DatLoader) Taining set. """

        self.testset = testset
        """ (torch.utils.data.DatLoader) Test set. """

        self.optimizer = optimizer
        """ (torch.optim.Optimizer) Optimizer. """

        self.scheduler = scheduler
        """ (torch.optim.LRScheduler) Scheduler. """

        self.augmentation = augmentation
        """ (imgaug.augmenters.Sequential) Augmentation. """

        self.cuda = cuda
        """ (bool) Run on CUDA. """

        self.loss = loss
        """ (callable) Classificaiton loss. """

        self.quantization = None
        """ (common.quantization.Quantization) Quantization. """

        self.projection = None
        """ (attacks.weights.projections.Projection) Projection, e.g., clipping. """

        self.fixed_quantization = False
        """ (bool) Fix quantization contexts. """

        self.fixed_quantization_contexts = None
        """ ([dict]) Quantization contexts. """

        self.summary_histograms = False
        """ (bool) Summary for histograms. """

        self.summary_weights = False
        """ (bool) Summary for weights. """

        self.summary_images = False
        """ (bool) Summary for images. """

        self.average = None
        """ (torch.nn.Module) Average. """

        self.keep_average = False
        """ (bool) Build average. """

        self.keep_average_tau = False
        """ (float) Average tau. """

        self.writer.add_text('config/host', common.utils.hostname())
        self.writer.add_text('config/pid', str(common.utils.pid()))
        self.writer.add_text('config/model', self.model.__class__.__name__)
        self.writer.add_text('config/model_details', str(self.model))
        self.writer.add_text('config/trainset', self.trainset.dataset.__class__.__name__)
        self.writer.add_text('config/testset', self.testset.dataset.__class__.__name__)
        self.writer.add_text('config/optimizer', self.optimizer.__class__.__name__)
        self.writer.add_text('config/scheduler', self.scheduler.__class__.__name__)
        self.writer.add_text('config/cuda', str(self.cuda))

        self.writer.add_text('model', str(self.model))
        self.writer.add_text('optimizer', str(common.summary.to_dict(self.optimizer)))
        self.writer.add_text('scheduler', str(common.summary.to_dict(self.scheduler)))

    def update_average(self):
        """
        Update average.
        """

        if self.keep_average:
            if self.average is None:
                self.average = common.torch.clone(self.model)
                self.average.eval()
                log('initialized average')
            else:
                average_parameters = list(self.average.parameters())
                parameters = list(self.model.parameters())

                for i in range(len(parameters)):
                    if parameters[i].requires_grad:
                        average_parameters[i].data = self.keep_average_tau*average_parameters[i].data + (1 - self.keep_average_tau)*parameters[i].data

                average_buffers = dict(self.average.named_buffers())
                backward_buffers = dict(self.model.named_buffers())

                for key in average_buffers.keys():
                    if key.find('num_batches_tracked') >= 0:
                        average_buffers[key].data += 1
                    if key.find('running_var') >= 0 or key.find('running_mean') >= 0:
                        average_buffers[key].data = self.keep_average_tau*backward_buffers[key].data + (1 - self.keep_average_tau)*backward_buffers[key].data

    def quantize(self):
        """
        Quantization.
        """

        if self.quantization is not None:
            forward_model, contexts = common.quantization.quantize(self.quantization, self.model, layers=self.layers, contexts=self.fixed_quantization_contexts)

            if self.fixed_quantization and self.fixed_quantization_contexts is None:
                assert self.fixed_quantization_contexts is None
                self.fixed_quantization_contexts = contexts
                log('fixed quantization contexts')

                # in case quantization is fixed, the bounds need to be enforced for the floating point model as well
                contexts_are_not_none = numpy.array([context is not None for context in contexts])
                if numpy.all(contexts_are_not_none):
                    min_bounds = []
                    max_bounds = []
                    for context in contexts:
                        if 'max_abs_range' in context.keys():
                            min_bounds.append(-context['max_abs_range'])
                            max_bounds.append(context['max_abs_range'])
                        if 'min_val' in context.keys():
                            min_bounds.append(-context['min_val'])
                        if 'max_val' in context.keys():
                            max_bounds.append(context['max_val'])
                    log('adding layer-wise projection:')
                    for i in range(len(min_bounds)):
                        log('\t%g - %g' % (min_bounds[i], max_bounds[i]))
                    projection = attacks.weights.projections.LayerWiseBoxProjection(min_bounds, max_bounds)
                    if self.projection is None:
                        self.projection = projection
                    else:
                        assert not isinstance(self.projection, attacks.weights.projections.SequentialProjections)
                        self.projection = attacks.weights.projections.SequentialProjections([self.projection, projection])

            assert self.fixed_quantization_contexts is None or self.fixed_quantization is True
            return forward_model, contexts
        return self.model, None

    def project(self):
        """
        Projection.
        """

        if self.projection is not None:
            self.projection(None, self.model, self.layers)

    def writer_add_model(self, contexts, global_step):
        """
        Monitor model.
        """

        j = 0
        for parameter in self.model.parameters():
            if parameter.requires_grad is False:  # normalization
                continue

            self.writer.add_scalar('train/max_weight/%d' % j, torch.max(parameter.data).item(), global_step=global_step)
            self.writer.add_scalar('train/q75_weight/%d' % j, common.torch.percentile(parameter.data, 75),
                                   global_step=global_step)
            self.writer.add_scalar('train/mean_weight/%d' % j, torch.mean(parameter.data).item(),
                                   global_step=global_step)
            self.writer.add_scalar('train/median_weight/%d' % j, torch.median(parameter.data).item(),
                                   global_step=global_step)
            self.writer.add_scalar('train/q25_weight/%d' % j, common.torch.percentile(parameter.data, 25),
                                   global_step=global_step)
            self.writer.add_scalar('train/min_weight/%d' % j, torch.min(parameter.data).item(),
                                   global_step=global_step)
            self.writer.add_scalar('train/mean_abs_weight/%d' % j, torch.mean(torch.abs(parameter.data)).item(),
                                   global_step=global_step)
            sum_abs_weight = torch.sum(torch.abs(parameter.data)).item()
            self.writer.add_scalar('train/sum_abs_weight/%d' % j, sum_abs_weight,
                                   global_step=global_step)
            self.writer.add_scalar('train/relevant_weight/%d' % j, sum_abs_weight / (torch.max(torch.abs(parameter.data)).item() * parameter.numel()),
                                   global_step=global_step)

            if contexts is not None:
                if contexts[j] is not None:
                    for key in ['max_abs_range', 'max_val', 'min_val', 'delta']:
                        if key in contexts[j].keys():
                            self.writer.add_scalar('train/quantization_%s/%d' % (key, j), contexts[j][key])

            if parameter.grad is not None:
                self.writer.add_scalar('train/max_gradient/%d' % j, torch.max(parameter.grad.data).item(),
                                       global_step=global_step)
                self.writer.add_scalar('train/q75_gradient/%d' % j, common.torch.percentile(parameter.grad.data, 75),
                                       global_step=global_step)
                self.writer.add_scalar('train/median_gradient/%d' % j, torch.median(parameter.grad.data).item(),
                                       global_step=global_step)
                self.writer.add_scalar('train/mean_gradient/%d' % j, torch.mean(parameter.grad.data).item(),
                                       global_step=global_step)
                self.writer.add_scalar('train/q25_gradient/%d' % j, common.torch.percentile(parameter.grad.data, 25),
                                       global_step=global_step)
                self.writer.add_scalar('train/min_gradient/%d' % j, torch.min(parameter.grad.data).item(),
                                       global_step=global_step)
                self.writer.add_scalar('train/mean_abs_gradient/%d' % j,
                                       torch.mean(torch.abs(parameter.grad.data)).item(),
                                       global_step=global_step)
                sum_abs_gradient = torch.sum(torch.abs(parameter.grad.data)).item()
                self.writer.add_scalar('train/sum_abs_gradient/%d' % j,
                                       sum_abs_gradient,
                                       global_step=global_step)
                self.writer.add_scalar('train/relevant_gradient/%d' % j,
                                       sum_abs_gradient / (torch.max(torch.abs(parameter.grad.data)).item() * parameter.numel()),
                                       global_step=global_step)
            if self.summary_histograms:
                self.writer.add_histogram('train/weights/%d' % j, parameter.view(-1), global_step=global_step)
                self.writer.add_histogram('train/gradients/%d' % j, parameter.grad.view(-1), global_step=global_step)
            j += 1

    def train(self, epoch):
        """
        Training step.

        :param epoch: epoch
        :type epoch: int
        """

        self.model.train()
        assert self.model.training is True

        # initialize contexts
        self.quantize()
        self.update_average()

        for b, (inputs, targets) in enumerate(self.trainset):
            if self.augmentation is not None:
                if isinstance(self.augmentation, iaa.meta.Augmenter):
                    inputs = self.augmentation.augment_images(inputs.numpy())
                else:
                    inputs = self.augmentation(inputs)

            inputs = common.torch.as_variable(inputs, self.cuda)
            inputs = inputs.permute(0, 3, 1, 2)
            assert len(targets.shape) == 1
            targets = common.torch.as_variable(targets, self.cuda)
            assert len(list(targets.size())) == 1

            self.project()
            forward_model, contexts = self.quantize()
            assert forward_model.training is True

            self.optimizer.zero_grad()
            logits = forward_model(inputs)
            loss = self.loss(logits, targets)
            error = common.torch.classification_error(logits, targets)
            loss.backward()

            if forward_model is not self.model:
                forward_parameters = list(forward_model.parameters())
                backward_parameters = list(self.model.parameters())

                for i in range(len(forward_parameters)):
                    if backward_parameters[i].requires_grad is False:  # normalization
                        continue

                    backward_parameters[i].grad = forward_parameters[i].grad

                forward_buffers = dict(forward_model.named_buffers())
                backward_buffers = dict(self.model.named_buffers())
                for key in forward_buffers.keys():
                    if key.find('running_var') >= 0 or key.find('running_mean') >= 0 or key.find('num_batches_tracked') >= 0:
                        backward_buffers[key].data = forward_buffers[key].data

            self.optimizer.step()
            self.scheduler.step()

            global_step = epoch * len(self.trainset) + b
            self.writer.add_scalar('train/lr', self.scheduler.get_lr()[0], global_step=global_step)
            self.writer.add_scalar('train/loss', loss.item(), global_step=global_step)
            self.writer.add_scalar('train/error', error.item(), global_step=global_step)
            self.writer.add_scalar('train/confidence', torch.mean(torch.max(common.torch.softmax(logits, dim=1), dim=1)[0]).item(), global_step=global_step)
            #print(loss.item(), error.item())

            if self.summary_histograms:
                self.writer.add_histogram('train/logits', torch.max(logits, dim=1)[0], global_step=global_step)
                self.writer.add_histogram('train/confidences', torch.max(common.torch.softmax(logits, dim=1), dim=1)[0], global_step=global_step)
            if self.summary_weights:
                self.writer_add_model(contexts, global_step)
            if self.summary_images:
                self.writer.add_images('train/images', inputs[:16], global_step=global_step)

            self.progress('train %d' % epoch, b, len(self.trainset), info='loss=%g error=%g lr=%g' % (
                loss.item(),
                error.item(),
                self.scheduler.get_lr()[0],
            ))

            self.update_average()

    def test(self, epoch):
        """
        Test step.

        :param epoch: epoch
        :type epoch: int
        """

        self.optimizer.zero_grad()
        self.model.eval()
        assert self.model.training is False

        self.project()
        forward_model, contexts = self.quantize()
        assert forward_model.training is False

        # reason to repeat this here: use correct loss for statistics
        losses = None
        errors = None
        logits = None
        probabilities = None

        for b, (inputs, targets) in enumerate(self.testset):
            if b >= 10:
                break;

            inputs = common.torch.as_variable(inputs, self.cuda)
            inputs = inputs.permute(0, 3, 1, 2)
            targets = common.torch.as_variable(targets, self.cuda)

            with torch.no_grad():
                outputs = forward_model(inputs)
                b_losses = self.loss(outputs, targets, reduction='none').detach().cpu().numpy()
                b_errors = common.torch.classification_error(outputs, targets, reduction='none').float().detach().cpu().numpy()

                losses = common.numpy.concatenate(losses, b_losses)
                errors = common.numpy.concatenate(errors, b_errors)
                logits = common.numpy.concatenate(logits, torch.max(outputs, dim=1)[0].detach().cpu().numpy())
                probabilities = common.numpy.concatenate(probabilities, common.torch.softmax(outputs, dim=1).detach().cpu().numpy())

                self.progress('test (test) %d' % epoch, b, len(self.testset), info='loss=%g error=%g' % (
                    numpy.mean(b_losses),
                    numpy.mean(b_errors)
                ))

        confidences = numpy.max(probabilities, axis=1)
        global_step = epoch  # epoch * len(self.trainset) + len(self.trainset) - 1

        self.writer.add_scalar('test/loss', numpy.mean(losses), global_step=global_step)
        self.writer.add_scalar('test/error', numpy.mean(errors), global_step=global_step)
        self.writer.add_scalar('test/logit', numpy.mean(logits), global_step=global_step)
        self.writer.add_scalar('test/confidence', numpy.mean(confidences), global_step=global_step)

        if self.summary_histograms:
            self.writer.add_histogram('test/losses', losses, global_step=global_step)
            self.writer.add_histogram('test/errors', errors, global_step=global_step)
            self.writer.add_histogram('test/logits', logits, global_step=global_step)
            self.writer.add_histogram('test/confidences', confidences, global_step=global_step)

        log('test (test): error=%g' % numpy.mean(errors))

        if self.average is not None:
            assert self.average.training is False
            losses = None
            errors = None
            logits = None
            probabilities = None

            for b, (inputs, targets) in enumerate(self.testset):
                if b >= 10:
                    break;

                inputs = common.torch.as_variable(inputs, self.cuda)
                inputs = inputs.permute(0, 3, 1, 2)
                targets = common.torch.as_variable(targets, self.cuda)

                with torch.no_grad():
                    outputs = self.average(inputs)
                    b_losses = self.loss(outputs, targets, reduction='none').detach().cpu().numpy()
                    b_errors = common.torch.classification_error(outputs, targets, reduction='none').float().detach().cpu().numpy()

                    losses = common.numpy.concatenate(losses, b_losses)
                    errors = common.numpy.concatenate(errors, b_errors)
                    logits = common.numpy.concatenate(logits, torch.max(outputs, dim=1)[0].detach().cpu().numpy())
                    probabilities = common.numpy.concatenate(probabilities, common.torch.softmax(outputs, dim=1).detach().cpu().numpy())

                    self.progress('test (test+average) %d' % epoch, b, len(self.testset), info='loss=%g error=%g' % (
                        numpy.mean(b_losses),
                        numpy.mean(b_errors)
                    ))

            confidences = numpy.max(probabilities, axis=1)
            global_step = epoch  # epoch * len(self.trainset) + len(self.trainset) - 1

            self.writer.add_scalar('test/average_loss', numpy.mean(losses), global_step=global_step)
            self.writer.add_scalar('test/average_error', numpy.mean(errors), global_step=global_step)
            self.writer.add_scalar('test/average_logit', numpy.mean(logits), global_step=global_step)
            self.writer.add_scalar('test/average_confidence', numpy.mean(confidences), global_step=global_step)

            if self.summary_histograms:
                self.writer.add_histogram('test/average_losses', losses, global_step=global_step)
                self.writer.add_histogram('test/average_errors', errors, global_step=global_step)
                self.writer.add_histogram('test/average_logits', logits, global_step=global_step)
                self.writer.add_histogram('test/average_confidences', confidences, global_step=global_step)

            log('test (test+average): error=%g' % numpy.mean(errors))

        return probabilities, forward_model, contexts