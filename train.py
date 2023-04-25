import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from time import time, strftime
import torch
import os
from tqdm import tqdm


def print_training_status(epoch_count, images_seen, train_loss, val_loss,
                          elapsed_time):
    print(
        # https://stackoverflow.com/a/8885688
        f"| {epoch_count:13.0f}",
        f"| {images_seen:13}",
        f"| {train_loss:13.3f}",
        f"| {val_loss:13.3f}",
        f"| {elapsed_time//60:10.0f}:{elapsed_time%60:02.0f} |",
    )


# TRAIN FUNCTION
def train_model(model, train_set, val_set, model_save_dir='./models/',
                 learning_rate=0.0005, max_epochs=30, patience=3,
                loss_fn=nn.CrossEntropyLoss(), is_autoencoder=False,
                optim_fn=torch.optim.Adam, batch_size=32, filename_note=None):

    # Init variables
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_name = (f'{model.__class__.__name__}-{model.num_classes}'
                       + ((filename_note + '_') if filename_note else '_')
                       + strftime("%d%m-%H%M") + '.pt')
    model_save_path = os.path.join(model_save_dir, model_save_name)
    optimizer = optim_fn(model.parameters(), learning_rate)
    images_seen = 0
    best_val_loss = None
    last_val_loss = None
    patience_count = 0
    halt_reason = None

    # move model to gpu for speed if available
    if torch.cuda.is_available():
        model.to('cuda')

    train_dataloader = DataLoader(train_set, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size, shuffle=True)

    print(f"""
TRAINING MODEL {model_save_name} WITH PARAMS:
 - Architecture: {model.__class__.__name__}
 - Learning rate: {learning_rate}
 - Optimizer: {optimizer.__class__.__name__}
 - Loss function: {loss_fn}
 - Other notes: {filename_note if filename_note else 'None'}

+---------------+---------------+---------------+---------------+---------------+
|         EPOCH | EXAMPLES SEEN |    TRAIN LOSS |      VAL LOSS |  ELAPSED TIME |
+---------------+---------------+---------------+---------------+---------------+
""", end='')

    start_time = time()  # Get start time to calculate training time later

    try: # For custom keyboard interrupt
        # TRAINING LOOP
        for epoch_count in range(1, max_epochs+1):
            # Train model with training set
            model.train()  # Set model to training mode
            train_loss = 0
            for images, labels in tqdm(train_dataloader, position=0, leave=False,
                                    desc=f'Epoch {epoch_count}') :  # iterate through batches
                if torch.cuda.is_available(): # can this be done to whole dataset instead?
                    images, labels = images.to('cuda'), labels.to('cuda')
                optimizer.zero_grad()
                outputs = model(images)  # Forward pass
                loss = loss_fn(outputs, labels) if isinstance(loss_fn, nn.CrossEntropyLoss) \
                    else loss_fn(outputs, labels.float()) / batch_size
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                images_seen += len(images)
            train_loss /= len(train_dataloader)  # Average loss over batch

            # Test model with separate validation set
            model.eval()  # Set model to evaluation mode
            val_loss = 0
            for images, labels in tqdm(val_dataloader, position=0, leave=False,
                                    desc=f'Testing epoch {epoch_count}'):
                if torch.cuda.is_available(): # can this be done to whole dataset instead?
                    images, labels = images.to('cuda'), labels.to('cuda')
                outputs = model(images)
                loss = loss_fn(outputs, labels) if isinstance(loss_fn, nn.CrossEntropyLoss) \
                    else loss_fn(outputs, labels.float()) / batch_size
                val_loss += loss.item()
            val_loss /= len(val_dataloader)  # Average loss over batch
            if best_val_loss is None or val_loss < best_val_loss:
                torch.save(model.state_dict(), model_save_path)
                best_val_loss = val_loss
            
            if last_val_loss and val_loss > last_val_loss:
                patience_count += 1
            else:
                # Can load best model to start again, or keep going in case it
                # starts improving again (uncomment):
                # model.load_state_dict(torch.load(model_save_path))
                patience_count = 0
            last_val_loss = val_loss

            # Display epoch results
            elapsed_time = time() - start_time
            print_training_status(epoch_count, images_seen, train_loss,
                                val_loss, elapsed_time)

            # Stop training if val loss hasn't improved for a while
            if patience_count >= patience:
                halt_reason = 'patience'
                break


    except KeyboardInterrupt:
        halt_reason = 'keyboard'
        pass
    
    print('+---------------+---------------+---------------+---------------+---------------+')
    
    if halt_reason is None:
        print(f"\nHlating training - epoch limit ({max_epochs}) reached")
    elif halt_reason == 'patience':
        print(f"\nHalting training - {patience} epochs without improvement")
    elif halt_reason == 'keyboard':
        print(f"\nHalting training - stopped by user")

    # Calculate total training time
    end_time = time()
    training_time_s = end_time - start_time

    print(
        f'\nTraining took {training_time_s//60:.0f}m {training_time_s%60:02.0f}s',
        f'({training_time_s//epoch_count}s per epoch)' if epoch_count > 0 else '')

    # Re-load best model from session to reverse overfitting
    if os.path.exists(model_save_path):
        print(f"\nBest model from session saved to '{model_save_path}'\n")
        model.load_state_dict(torch.load(model_save_path))
    else:
        print('Model NOT saved - check previous error messages')

    # Return best model
    return model