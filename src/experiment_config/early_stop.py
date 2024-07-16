# The function checkBest is called after every validation. It is responsible for stopping the code if it doesn't improve for too long.

from hyper_parameters import EPOCHS, PATIENCE
import os

no_improve = 0
prev_epoch = 0
best_model = {"epoch": 0, "loss": -1}

def checkBest(self, runner):
    global best_model, prev_epoch, no_improve
    
    crr_epoch = runner.epoch
    crr_losses = runner.log_buffer.val_history["loss_val"]
    crr_loss = sum(crr_losses) / len(crr_losses)  # mean loss value.

    if crr_epoch != prev_epoch:  # this validation is due to this function being called 5 times at once.
        prev_epoch = crr_epoch

        if crr_epoch == 1:  # first iteration
            best_model = {
                "epoch": crr_epoch,
                "loss": crr_loss
            }
        else:
            if best_model["loss"] >= crr_loss:  # update the best_model if the loss has been improved
                best_model = {
                    "epoch": crr_epoch,
                    "loss": crr_loss
                }
                no_improve = 0
                epoch_to_remove = f"epoch_{best_model['epoch']}.pth"  # remove previous best
            else:  # if the loss doesn't improve
                no_improve += 1
                epoch_to_remove = f"epoch_{crr_epoch}.pth"  # remove last epoch

            os.remove(os.path.join(runner.work_dir, epoch_to_remove))

            if no_improve > PATIENCE * EPOCHS:
                os.remove(os.path.join(runner.work_dir, "latest.pth"))
                os.rename(
                    os.path.join(runner.work_dir, f'epoch_{best_model["epoch"]}.pth'),
                    os.path.join(runner.work_dir, 'latest.pth')
                )  # rename the best epoch as the latest, in order to standardize the name

                no_improve = 0
                prev_epoch = 0
                best_model = None

                raise InterruptedError("end training")  # it crashes the code; the exception will be caught later

            print(f'Epochs without improvement: {no_improve}')
