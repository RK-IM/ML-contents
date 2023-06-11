from utils import calculate_metrics

from tqdm import tqdm
from tabulate import tabulate

import torch

# Train
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.Module,
               device: torch.device):
    train_loss = 0
    pred_probs = []
    true_labels = []

    model.train()
    for X, y in tqdm(dataloader):
        # set device
        X = X.to(device)
        y = y.to(device)

        # forward pass
        outputs = model(X)

        # calculate loss
        loss = loss_fn(outputs, y)

        # loss backward, optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate results
        train_loss += loss.item()
        pred_probs.append(torch.sigmoid(outputs).cpu())
        true_labels.append(y.cpu().type(torch.IntTensor))

    # update results
    train_loss /= len(dataloader)
    pred_probs = torch.cat(pred_probs, axis=0)
    true_labels = torch.cat(true_labels, axis=0)

    results = calculate_metrics(pred_probs, true_labels)
    return train_loss, results


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module, 
              device: torch.device):
    test_loss = 0
    test_pred_probs = []
    test_true_labels = []

    model.eval()
    with torch.inference_mode():
        for X_test, y_test in tqdm(dataloader):
            # Set device
            X_test, y_test = X_test.to(device), y_test.to(device)

            # forward step
            test_outputs = model(X_test)

            # calculate loss
            loss = loss_fn(test_outputs, y_test)
            test_loss += loss.item()
            
            test_pred_probs.append(torch.sigmoid(test_outputs).cpu())
            test_true_labels.append(y_test.cpu().type(torch.IntTensor))

    test_loss /= len(dataloader)
    test_pred_probs = torch.cat(test_pred_probs, axis=0)
    test_true_labels = torch.cat(test_true_labels, axis=0)
    
    results = calculate_metrics(test_pred_probs, test_true_labels)
    
    return test_loss, results


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer, 
          loss_fn: torch.nn.Module, 
          writer: torch.utils.tensorboard.SummaryWriter, 
          device: torch.device, 
          epochs: int,
          patience: int = 3,):
    count = 1
    threshold = 1e9

    for epoch in range(epochs):
        train_loss, train_results = train_step(
            model=model, 
            dataloader=train_dataloader, 
            optimizer=optimizer,
            loss_fn=loss_fn, 
            device=device)
        test_loss, test_results = test_step(
            model=model, 
            dataloader=test_dataloader, 
            loss_fn=loss_fn, 
            device=device)
        
        # display.clear_output(wait=True)
        
        print(f"\nEpochs: {epoch+1} | "
              f"Loss: {train_loss:.4f} | "
              f"Valid Loss: {test_loss:.4f}")
        
        print("Train Results:")
        print(tabulate(train_results, headers='keys', tablefmt='psql'))
        # display.display(train_results)
        print("\nTest Results:")
        print(tabulate(test_results, headers='keys', tablefmt='psql'))
        # display.display(test_results)
        
        if writer:
            # loss
            writer.add_scalars(main_tag="Loss",
                               tag_scalar_dict={'train_loss': train_loss,
                                               'test_loss': test_loss},
                               global_step=epoch)

            # metrics
            writer.add_scalars(main_tag="train/F1_score",
                               tag_scalar_dict=train_results.loc[:, 'F1 score'].to_dict(),
                               global_step=epoch)
            writer.add_scalars(main_tag="train/ROC_AUC",
                               tag_scalar_dict=train_results.loc[:, 'ROC AUC'].to_dict(),
                               global_step=epoch)
            writer.add_scalars(main_tag="train/PR_AUC",
                               tag_scalar_dict=train_results.loc[:, 'PR AUC'].to_dict(),
                               global_step=epoch)

            writer.add_scalars(main_tag="test/F1_score",
                               tag_scalar_dict=test_results.loc[:, 'F1 score'].to_dict(),
                               global_step=epoch)
            writer.add_scalars(main_tag="test/ROC_AUC",
                               tag_scalar_dict=test_results.loc[:, 'ROC AUC'].to_dict(),
                               global_step=epoch)
            writer.add_scalars(main_tag="test/PR_AUC",
                               tag_scalar_dict=test_results.loc[:, 'PR AUC'].to_dict(),
                               global_step=epoch)
        
        else:
            pass
        
        if test_loss <= threshold:
            threshold = test_loss
            model.best_params = model.state_dict()
        else:
            if count >= patience:
                print(f"[INFO] Model doesn't improved in {patience} epochs")
                break
            else:
                count += 1
    if writer:
        writer.close()