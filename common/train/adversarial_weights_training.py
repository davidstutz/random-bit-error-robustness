import numpy
import torch
import common.torch
import common.summary
import common.numpy
import attacks.weights
from imgaug import augmenters as iaa
from .normal_training import NormalTraining


class AdversarialWeightsTraining(NormalTraining):
    """
    Adversarial weights training.
    """

    def __init__(self, model, trainset, testset, optimizer, scheduler, attack, objective, operators=None, augmentation=None, loss=common.torch.classification_loss, writer=common.summary.SummaryWriter(), cuda=False):
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
        :param attack: attack
        :type attack: attacks.Attack
        :param objective: objective
        :type objective: attacks.Objective
        :param augmentation: augmentation
        :type augmentation: imgaug.augmenters.Sequential
        :param writer: summary writer
        :type writer: torch.utils.tensorboard.SummaryWriter or TensorboardX equivalent
        :param cuda: run on CUDA device
        :type cuda: bool
        """

        assert isinstance(attack, attacks.weights.Attack)
        assert getattr(attack, 'training', None) is not None
        assert isinstance(objective, attacks.weights.objectives.Objective)
        assert getattr(attack, 'norm', None) is not None

        super(AdversarialWeightsTraining, self).__init__(model, trainset, testset, optimizer, scheduler, augmentation, loss, writer, cuda)

        self.attack = attack
        """ (attacks.weights.Attack) Attack. """

        self.objective = objective
        """ (attacks.weights.Objective) Objective. """

        self.max_batches = 10
        """ (int) Number of batches to test adversarially on. """

        self.curriculum = None
        """ (None or callable) Curriculum for attack. """

        self.population = 0
        """ (int) Population. """

        self.gradient_clipping = 0.05
        """ (float) Clipping. """

        self.reset_iterations = 1
        """ (int) Reset objective iterations. """

        self.average_statistics = False
        """ (bool) Average bn statistics. """

        self.adversarial_statistics = False
        """ (bool) Adversarial bn statistics. """

        self.operators = operators
        """ ([attacks.activations.operator.Operator]) Operators. """

        self.attack.training = True
        self.writer.add_text('config/attack', self.attack.__class__.__name__)
        self.writer.add_text('config/objective', self.objective.__class__.__name__)
        self.writer.add_text('attack', str(common.summary.to_dict(self.attack)))
        if getattr(attack, 'initialization', None) is not None:
            self.writer.add_text('attack/initialization', str(common.summary.to_dict(self.attack.initialization)))
            if getattr(self.attack.initialization, 'initializations', None) is not None:
                for i in range(len(self.attack.initialization.initializations)):
                    self.writer.add_text('attack/initialization_%d' % i, str(common.summary.to_dict(self.attack.initialization.initializations[i])))
        if getattr(attack, 'projection', None) is not None:
            self.writer.add_text('attack/projection', str(common.summary.to_dict(self.attack.projection)))
            if getattr(self.attack.projection, 'projections', None) is not None:
                for i in range(len(self.attack.projection.projections)):
                    self.writer.add_text('attack/projection_%d' % i, str(common.summary.to_dict(self.attack.projection.projections[i])))
        if getattr(attack, 'norm', None) is not None:
            self.writer.add_text('attack/norm', str(common.summary.to_dict(self.attack.norm)))
        self.writer.add_text('objective', str(common.summary.to_dict(self.objective)))

    def writer_add_perturbed_model(self, perturbed_model,i, global_step):
        """
        Monitor perturbed model.
        """

        parameters = list(self.model.parameters())
        perturbed_parameters = list(perturbed_model.parameters())

        for j in range(len(parameters)):
            if parameters[j].requires_grad is False:  # normalization
                continue

            parameter = parameters[j]
            perturbed_parameter = perturbed_parameters[j]

            self.writer.add_scalar('train/mean_weight/%d' % j, torch.mean(perturbed_parameter.data).item(),
                                   global_step=global_step)
            self.writer.add_scalar('train/max_weight/%d' % j, torch.max(perturbed_parameter.data).item(), global_step=global_step)
            self.writer.add_scalar('train/q75_weight/%d' % j, common.torch.percentile(perturbed_parameter.data, 75),
                                   global_step=global_step)
            self.writer.add_scalar('train/median_weight/%d' % j, torch.median(perturbed_parameter.data).item(),
                                   global_step=global_step)
            self.writer.add_scalar('train/q25_weight/%d' % j, common.torch.percentile(perturbed_parameter.data, 25),
                                   global_step=global_step)
            self.writer.add_scalar('train/min_weight/%d' % j, torch.min(perturbed_parameter.data).item(),
                                   global_step=global_step)
            self.writer.add_scalar('train/mean_abs_weight/%d' % j, torch.mean(torch.abs(perturbed_parameter.data)).item(),
                                   global_step=global_step)
            self.writer.add_scalar('train/sum_abs_weight/%d' % j, torch.sum(torch.abs(perturbed_parameter.data)).item(),
                                   global_step=global_step)
            self.writer.add_scalar('train/positive_weight/%d' % j,
                                   torch.sum(perturbed_parameter.data > 1e-6).item() / perturbed_parameter.numel(), global_step=global_step)
            self.writer.add_scalar('train/negative_weight/%d' % j,
                                   torch.sum(perturbed_parameter.data < -1e-6).item() / perturbed_parameter.numel(),
                                   global_step=global_step)
            self.writer.add_scalar('train/relevant_weight/%d' % j,
                                   torch.sum(torch.abs(perturbed_parameter.data)).item() / (torch.max(torch.abs(perturbed_parameter.data)).item() * perturbed_parameter.numel()),
                                   global_step=global_step)

            self.writer.add_scalar('train/mean_abs_weight_difference/%d' % j, torch.mean(torch.abs(parameter.data - perturbed_parameter.data)).item(),
                                   global_step=global_step)
            self.writer.add_scalar('train/max_abs_weight_difference/%d' % j, torch.max(torch.abs(parameter.data - perturbed_parameter.data)).item(),
                                   global_step=global_step)
            self.writer.add_scalar('train/sum_abs_weight_difference/%d' % j, torch.sum(torch.abs(parameter.data - perturbed_parameter.data)).item(),
                                   global_step=global_step)
            self.writer.add_scalar('train/relative_abs_weight_difference/%d' % j,
                                   torch.sum(torch.abs(parameter.data - perturbed_parameter.data)).item() / torch.sum(torch.abs(parameter.data)).item(),
                                   global_step=global_step)

            if perturbed_parameter.grad is not None:
                self.writer.add_scalar('train/mean_gradient/%d' % j, torch.mean(perturbed_parameter.grad.data).item(),
                                       global_step=global_step)
                self.writer.add_scalar('train/max_gradient/%d' % j, torch.max(perturbed_parameter.grad.data).item(),
                                       global_step=global_step)
                self.writer.add_scalar('train/q75_gradient/%d' % j, common.torch.percentile(perturbed_parameter.grad.data, 75),
                                       global_step=global_step)
                self.writer.add_scalar('train/median_gradient/%d' % j, torch.median(perturbed_parameter.grad.data).item(),
                                       global_step=global_step)
                self.writer.add_scalar('train/q25_gradient/%d' % j, common.torch.percentile(perturbed_parameter.grad.data, 25),
                                       global_step=global_step)
                self.writer.add_scalar('train/min_gradient/%d' % j, torch.min(perturbed_parameter.grad.data).item(),
                                       global_step=global_step)
                self.writer.add_scalar('train/mean_abs_gradient/%d' % j,
                                       torch.mean(torch.abs(perturbed_parameter.grad.data)).item(),
                                       global_step=global_step)
                self.writer.add_scalar('train/sum_abs_gradient/%d' % j,
                                       torch.sum(torch.abs(perturbed_parameter.grad.data)).item(),
                                       global_step=global_step)
                self.writer.add_scalar('train/positive_gradient/%d' % j,
                                       torch.sum(perturbed_parameter.grad.data > 1e-6).item() / perturbed_parameter.numel(),
                                       global_step=global_step)
                self.writer.add_scalar('train/negative_gradient/%d' % j,
                                       torch.sum(perturbed_parameter.grad.data < -1e-6).item() / perturbed_parameter.numel(),
                                       global_step=global_step)
                self.writer.add_scalar('train/relevant_gradient/%d' % j,
                                       torch.sum(torch.abs(perturbed_parameter.grad.data)).item() / (torch.max(torch.abs(perturbed_parameter.grad.data)).item() * perturbed_parameter.numel()),
                                       global_step=global_step)
            if self.summary_histograms:
                self.writer.add_histogram('train/weights/%d' % j, perturbed_parameter.view(-1), global_step=global_step)
                self.writer.add_histogram('train/gradients/%d' % j, perturbed_parameter.grad.view(-1), global_step=global_step)

    def train(self, epoch):
        """
        Training step.

        :param epoch: epoch
        :type epoch: int
        """

        assert self.average_statistics is False
        assert not (self.average_statistics and self.adversarial_statistics)

        # initialize contexts
        self.quantize()

        for b, (inputs, targets) in enumerate(self.trainset):
            if self.augmentation is not None:
                if isinstance(self.augmentation, iaa.meta.Augmenter):
                    inputs = self.augmentation.augment_images(inputs.numpy())
                    print('augmented')
                else:
                    inputs = self.augmentation(inputs)

            # before permutation!
            # works with enumerate() similar to data loader.
            batchset = [(inputs, targets)]

            inputs = common.torch.as_variable(inputs, self.cuda)
            inputs = inputs.permute(0, 3, 1, 2)
            targets = common.torch.as_variable(targets, self.cuda)

            self.project()
            forward_model, contexts = self.quantize()
            global_step = epoch * len(self.trainset) + b

            population_norm = 0
            population_perturbed_loss = 0
            population_perturbed_error = 0

            mean_abs_grad = 0
            if self.population > 0:
                self.model.train()
                forward_model.train()
                self.optimizer.zero_grad()
                self.model.zero_grad()
                forward_model.zero_grad()

                logits = forward_model(inputs)

                loss = self.loss(logits, targets)
                error = common.torch.classification_error(logits, targets)

                if not self.adversarial_statistics:
                    forward_buffers = dict(forward_model.named_buffers())
                    backward_buffers = dict(self.model.named_buffers())
                    for key in forward_buffers.keys():
                        if key.find('running_var') >= 0 or key.find('running_mean') >= 0 or key.find('num_batches_tracked') >= 0:
                            backward_buffers[key].data = forward_buffers[key].data

                self.model.eval()
                forward_model.eval()
                self.optimizer.zero_grad()
                self.model.zero_grad()
                forward_model.zero_grad()

                for i in range(self.population):
                    self.objective.reset()
                    perturbed_model = self.attack.run(forward_model, batchset, self.objective)

                    # This is a perturbation based on the original eval model (self.model.eval)!
                    perturbed_model.train()
                    perturbed_model.zero_grad()
                    perturbed_logits = perturbed_model(inputs, operators=self.operators)

                    perturbed_loss = self.loss(perturbed_logits, targets)
                    perturbed_error = common.torch.classification_error(perturbed_logits, targets)

                    perturbed_loss.backward()

                    population_perturbed_loss += perturbed_loss.item()
                    population_perturbed_error += perturbed_error.item()

                    # take average of gradients
                    parameters = list(self.model.parameters())
                    perturbed_parameters = list(perturbed_model.parameters())

                    norm_ = 0
                    perturbed_norm_ = 0
                    for j in range(len(parameters)):
                        norm_ = max(norm_, torch.abs(torch.max(parameters[j].data)).item())
                        perturbed_norm_ = max(perturbed_norm_, torch.abs(torch.max(parameters[j].data - perturbed_parameters[j].data)))

                        if parameters[j].requires_grad is False:  # normalization
                            continue

                        if parameters[j].grad is None:
                            parameters[j].grad = torch.clamp(perturbed_parameters[j].grad, min=-self.gradient_clipping, max=self.gradient_clipping)
                        else:
                            parameters[j].grad.data += torch.clamp(perturbed_parameters[j].grad.data, min=-self.gradient_clipping, max=self.gradient_clipping)

                    if self.adversarial_statistics:
                        perturbed_buffers = dict(perturbed_model.named_buffers())
                        backward_buffers = dict(self.model.named_buffers())
                        for key in perturbed_buffers.keys():
                            if key.find('running_var') >= 0 or key.find('running_mean') >= 0 or key.find('num_batches_tracked') >= 0:
                                if i == 0:
                                    backward_buffers[key].data.fill_(0)
                                backward_buffers[key].data += perturbed_buffers[key].data

                    if self.summary_weights:
                        self.writer_add_perturbed_model(perturbed_model, i, global_step)

                    self.writer.add_scalar('train/adversarial_loss%d' % i, perturbed_loss.item(), global_step=global_step)
                    self.writer.add_scalar('train/adversarial_error%d' % i, perturbed_error.item(), global_step=global_step)

                    if self.attack.norm is not None:
                        norm = self.attack.norm(forward_model, perturbed_model, self.attack.layers, self.quantization, contexts)
                        population_norm += norm
                        self.writer.add_scalar('train/adversarial_norm%d' % i, norm, global_step=global_step)
                        for j in range(len(self.attack.norm.norms)):
                            self.writer.add_scalar('train/adversarial_norms%d/%d' % (i, j), self.attack.norm.norms[j], global_step=global_step)

                population_norm /= self.population
                population_perturbed_loss /= self.population
                population_perturbed_error /= self.population

                for parameter in self.model.parameters():
                    if parameter.requires_grad is False:  # normalization
                        continue
                    parameter.grad.data /= self.population
                    mean_abs_grad += torch.mean(torch.abs(parameter.grad.data))
                mean_abs_grad /= len(list(self.model.parameters()))

                if self.adversarial_statistics:
                    backward_buffers = dict(self.model.named_buffers())
                    for key in backward_buffers.keys():
                        if key.find('running_var') >= 0 or key.find('running_mean') >= 0:
                            backward_buffers[key].data /= self.population
                        elif key.find('num_batches_tracked') >= 0:
                            backward_buffers[key].data //= self.population

                self.model.train()
                forward_model.train()

                self.optimizer.step()
                self.scheduler.step()
            else:
                # basically normal training
                self.model.train()
                forward_model.train()
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

                    # take care of BN statistics
                    forward_buffers = dict(forward_model.named_buffers())
                    backward_buffers = dict(self.model.named_buffers())
                    for key in forward_buffers.keys():
                        if key.find('running_var') >= 0 or key.find('running_mean') >= 0:
                            backward_buffers[key].data = forward_buffers[key].data
                        elif key.find('num_batches_tracked') >= 0:
                            backward_buffers[key].data //= self.population

                self.optimizer.step()
                self.scheduler.step()

            curriculum_logs = dict()
            if self.curriculum is not None:
                self.population, curriculum_logs = self.curriculum(self.attack, loss, population_perturbed_loss, epoch)
                for curriculum_key, curriculum_value in curriculum_logs.items():
                    self.writer.add_scalar('train/curriculum/%s' % curriculum_key, curriculum_value, global_step=global_step)
            else:
                self.population = 1

            self.writer.add_scalar('train/lr', self.scheduler.get_lr()[0], global_step=global_step)

            self.writer.add_scalar('train/loss', loss.item(), global_step=global_step)
            self.writer.add_scalar('train/error', error.item(), global_step=global_step)
            self.writer.add_scalar('train/confidence', torch.mean(torch.max(common.torch.softmax(logits, dim=1), dim=1)[0]).item(), global_step=global_step)

            if self.summary_histograms:
                self.writer.add_histogram('train/logits', torch.max(logits, dim=1)[0], global_step=global_step)
                self.writer.add_histogram('train/confidences', torch.max(common.torch.softmax(logits, dim=1), dim=1)[0], global_step=global_step)
            if self.summary_weights:
                self.writer_add_model(contexts, global_step)
            if self.summary_images:
                self.writer.add_images('train/images', inputs[:16], global_step=global_step)

            self.progress('train %d' % epoch, b, len(self.trainset), info='loss=%g err=%g advloss=%g adverr=%g advnorm=%g gradient=%g lr=%g pop=%d curr=%s' % (
                loss.item(),
                error.item(),
                population_perturbed_loss,
                population_perturbed_error,
                population_norm,
                mean_abs_grad,
                self.scheduler.get_lr()[0],
                self.population,
                str(list(curriculum_logs.values())),
            ))

    def test(self, epoch):
        """
        Test on adversarial examples.

        :param epoch: epoch
        :type epoch: int
        """

        probabilities, forward_model, contexts = super(AdversarialWeightsTraining, self).test(epoch)

        assert forward_model.training is False
        assert self.model.training is False
        losses = None
        errors = None
        logits = None
        confidences = None
        norms = []

        if getattr(self.attack, 'error_bound', None) is not None:
            error_bound = self.attack.error_bound
            self.attack.error_bound = -1e12

        for b, (inputs, targets) in enumerate(self.testset):
            if b >= self.max_batches:
                break

            batchset = [(inputs, targets)]
            self.objective.reset()
            perturbed_model = self.attack.run(forward_model, batchset, self.objective)

            inputs = common.torch.as_variable(inputs, self.cuda)
            inputs = inputs.permute(0, 3, 1, 2)
            targets = common.torch.as_variable(targets, self.cuda)

            perturbed_model.eval()
            with torch.no_grad():
                outputs = perturbed_model(inputs, operators=self.operators)
                b_losses = self.loss(outputs, targets, reduction='none')
                b_errors = common.torch.classification_error(outputs, targets, reduction='none')

                losses = common.numpy.concatenate(losses, b_losses.detach().cpu().numpy())
                errors = common.numpy.concatenate(errors, b_errors.detach().cpu().numpy())
                logits = common.numpy.concatenate(logits, torch.max(outputs, dim=1)[0].detach().cpu().numpy())
                confidences = common.numpy.concatenate(confidences, torch.max(common.torch.softmax(outputs, dim=1), dim=1)[0].detach().cpu().numpy())
                if self.attack.norm is not None:
                    norms.append(self.attack.norm(forward_model, perturbed_model, self.attack.layers, self.quantization, contexts))

                self.progress('test %d' % epoch, b, self.max_batches, info='loss=%g error=%g' % (
                    torch.mean(b_losses).item(),
                    torch.mean(b_errors.float()).item()
                ))

        global_step = epoch
        self.writer.add_scalar('test/adversarial_loss', numpy.mean(losses), global_step=global_step)
        self.writer.add_scalar('test/adversarial_error', numpy.mean(errors), global_step=global_step)
        self.writer.add_scalar('test/adversarial_logit', numpy.mean(logits), global_step=global_step)
        self.writer.add_scalar('test/adversarial_confidence', numpy.mean(confidences), global_step=global_step)

        norms = numpy.array(norms)
        self.writer.add_scalar('test/adversarial_norm', numpy.mean(norms), global_step=global_step)

        if self.summary_histograms:
            self.writer.add_histogram('test/adversarial_losses', losses, global_step=global_step)
            self.writer.add_histogram('test/adversarial_errors', errors, global_step=global_step)
            self.writer.add_histogram('test/adversarial_logits', logits, global_step=global_step)
            self.writer.add_histogram('test/adversarial_confidences', confidences, global_step=global_step)
            self.writer.add_histogram('test/adversarial_norms', norms, global_step=global_step)

        if getattr(self.attack, 'error_bound', None) is not None:
            self.attack.error_bound = error_bound

        return probabilities, forward_model, contexts