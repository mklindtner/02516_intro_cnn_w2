import logging
from time import time


from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


LOG = logging.getLogger(__name__)


def train(model, opt, loss_fn, epochs, train_loader, test_loader, device):
    X_test, Y_test = next(iter(test_loader))

    for epoch in range(epochs):
        tic = time()
        LOG.info('* Epoch test %d/%d' % (epoch+1, epochs))
        
        avg_loss = 0
        model.train()  # train mode
        for X_batch, Y_batch in train_loader:
            LOG.debug("here")
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            LOG.debug(f'shape x_batch before model: {X_batch.shape}')
            LOG.debug(f'shape y_batch: {Y_batch.shape}')
            opt.zero_grad()
 
            Y_pred = model(X_batch)
            LOG.debug(f'X_batch shape after model: {Y_pred.shape}')
            loss = loss_fn(Y_pred, Y_batch)  
            loss.backward() 
            opt.step()  

            avg_loss += loss / len(train_loader)

        model.eval()  
        val_loss = 0
        dice_loss_val, iou_loss_val, acc_val, sens_val, spec_val = 0, 0, 0, 0, 0
        with torch.no_grad():  
            for X_val, Y_val in val_loader:
                X_val = X_val.to(device)
                Y_val = Y_val.to(device)

                Y_pred_val = model(X_val)
                loss_val = loss_fn(Y_pred_val, Y_val)
                val_loss += loss_val.item() * X_val.size(0)


                Y_pred_binary = torch.round(Y_pred_val)

                dice_loss_val += dice_coefficient(Y_pred_binary, Y_val)
                iou_loss_val += iou_loss(Y_pred_binary, Y_val)
                acc_val += accuracy(Y_pred_binary, Y_val)
                sens_val += sensitivity(Y_pred_binary, Y_val)
                spec_val += specificity(Y_pred_binary, Y_val)

        # Average out the metrics over the entire validation dataset
        num_batches = len(val_loader)
        val_loss /= len(val_loader.dataset)
        dice_loss_val /= num_batches
        iou_loss_val /= num_batches
        acc_val /= num_batches
        sens_val /= num_batches
        spec_val /= num_batches

        toc = time()
        LOG.info(' - loss: %f' % avg_loss)
       
        Y_hat = F.sigmoid(model(X_test.to(device))).detach().cpu()
        clear_output(wait=True)
        for k in range(6):
            plt.subplot(2, 6, k+1)
            plt.imshow(np.rollaxis(X_test[k].numpy(), 0, 3), cmap='gray')
            plt.title('Real')
            plt.axis('off')

            plt.subplot(2, 6, k+7)
            plt.imshow(Y_hat[k, 0], cmap='gray')
            plt.title('Output')
            plt.axis('off')
        plt.suptitle('%d / %d - loss: %f' % (epoch+1, epochs, avg_loss))
        plt.show()
    return dice_loss_val.item(), iou_loss_val.item(), acc_val.item(), sens_val.item(), spec_val.item()