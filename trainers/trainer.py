
class Trainer(object):
    model = None
    best_state = None

    def __init__(self, model, criterion, args, optimizer):
        pass

    def train(self, train_loader, model, criterion, optimizer, epoch):
        raise NotImplementedError

    def valid(self, val_loader, model, criterion, optimizer, epoch):
        raise NotImplementedError
