from keras.callbacks import Callback

class CustomVerboseCallback(Callback):
    def __init__(self, total_epochs, print_every=10, progress_callback=None, start_progress=0.0, end_progress=1.0):
        super().__init__()
        self.total_epochs = total_epochs
        self.print_every = print_every
        self.progress_callback = progress_callback
        self.start_progress = start_progress
        self.end_progress = end_progress

    def on_epoch_end(self, epoch, logs=None):
        epoch_1based = epoch + 1
        if epoch_1based % self.print_every == 0 or epoch_1based == self.total_epochs:
            loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)
            print(f"Epoch {epoch_1based}/{self.total_epochs} - loss: {loss:.4f} - val_loss: {val_loss:.4f}")
        
        if self.progress_callback:
            # Map epoch progress (0 to 1) to (start_progress to end_progress)
            current_progress = self.start_progress + (epoch_1based / self.total_epochs) * (self.end_progress - self.start_progress)
            self.progress_callback(current_progress)
