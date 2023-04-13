import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from time import time, strftime
import torch
import os
import tests


def save_net(model, path):
    try:
        torch.save(model.state_dict(), path)
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    except Exception as e:
        print(f"Issue saving model to {path}: {e}")


def print_training_status(epoch_count, images_seen, train_loss, val_loss, elapsed_time):
    print(
        # https://stackoverflow.com/a/8885688
        f"{epoch_count:5.0f} |",
        f"{images_seen:13} |",
        f"{train_loss:10.2f} |",
        f"{val_loss:8.2f} |",
        f"{elapsed_time//60:9.0f}:{elapsed_time%60:02.0f}",
    )


# TRAIN FUNCTION
def train_model(model, train_set, val_set, model_save_dir='../models',
                 learning_rate=0.001, max_epochs=30, patience=3,
                loss_fn=nn.CrossEntropyLoss(), is_autoencoder=False,
                optim_fn=torch.optim.SGD, batch_size=64, filename_note=None):
    
    # Init variables
    model_save_name = (f'{model.__class__.__name__}'
                       + ((filename_note + '_') if filename_note else '')
                       + strftime("%d-%m-%H-%M") + '.pt')
    model_save_path = os.path.join(model_save_dir, model_save_name)
    optimizer = optim_fn(model.parameters(), learning_rate)
    epoch_count = 0
    images_seen = 0
    best_val_loss = None
    val_loss_history = []

    train_dataloader = DataLoader(train_set, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size, shuffle=True)

    print()
    print(f"TRAINING MODEL {model_save_name} WITH PARAMS:")
    print(f" - Architecture: {model.__class__.__name__}")
    print(f" - Learning rate: {learning_rate}")
    print(f" - Optimizer: {optimizer.__class__.__name__}")
    print(f" - Loss function: {loss_fn}")
    print(f" - Other notes: None" if not filename_note else f" - Other notes: {filename_note}")
    print()
    print("EPOCH | EXAMPLES SEEN | TRAIN LOSS | VAL LOSS | ELAPSED TIME")

    start_time = time()  # Get start time to calculate training time later

    try: # For custom keyboard interrupt
        # TRAINING LOOP
        for epoch_count in range(max_epochs):
            # Train model with training set
            model.train() # Set model to training mode
            train_loss = 0
            for images, labels in train_dataloader:  # iterate through batches
                # Forward pass
                outputs = model(images)
                if is_autoencoder:
                    labels = images
                loss = loss_fn(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                images_seen += len(images)
            train_loss /= len(train_dataloader) # Average loss over batch

            # Test model with separate validation set
            model.eval() # Set model to evaluation mode
            val_loss = 0
            for images, labels in val_dataloader:
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
            val_loss /= len(val_dataloader) # Average loss over batch
            if best_val_loss is None or val_loss < best_val_loss:
                save_net(model, model_save_path)
                best_val_loss = val_loss

            # Display epoch results
            elapsed_time = time() - start_time
            print_training_status(epoch_count, images_seen, train_loss,
                                    val_loss, elapsed_time)

            # Stop training if val loss hasn't improved for a while
            if epoch_count >= patience and val_loss >= val_loss_history[-patience]:
                print(f"\nHalting training - {patience} epochs without improvement")
                break

            val_loss_history.append(val_loss)

    except KeyboardInterrupt:
        print(f"\nHalting training - keyboard interrupt")
        pass

    # Calculate total training time
    end_time = time()
    training_time_s = end_time - start_time

    print(
        f'\nTraining took {training_time_s//60:.0f}m {training_time_s%60:02.0f}s',
        f'({training_time_s//epoch_count}s per epoch)')

    print(f"\nBest model from session saved to '{model_save_path}'\n")
    
    # Return trained model
    return model