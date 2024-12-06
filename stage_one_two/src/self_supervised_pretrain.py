import os
import torch
from monai.losses import ContrastiveLoss
from monai.networks.nets import ViTAutoEnc
from load_data import get_unlabeled_data_loader, get_labeled_data_loader, self_sup_transforms
from stage_one_two.utils import plot_losses

# Define the Self-Supervised ViT Autoencoder
model = ViTAutoEnc(
    in_channels=1,
    img_size=(96, 96, 96),
    patch_size=(16, 16, 16),
    proj_type="conv",
    hidden_size=768,
    mlp_dim=3072,
)
# TODO loading pretrain weights
pretrain_path = 'F://open_project//FLARE//outputs_self//best_model.pth'
model.load_state_dict(torch.load(pretrain_path))

def train(model, loader, optimizer, recon_loss, contrastive_loss, device):
    model.train()
    epoch_loss, epoch_cl_loss, epoch_recon_loss = 0, 0, 0
    step = 0

    for batch_data in loader:
        step += 1
        inputs, inputs_2, gt_input = (
            batch_data["image"].to(device),
            batch_data["image_2"].to(device),
            batch_data["gt_image"].to(device),
        )

        optimizer.zero_grad()
        outputs_v1, hidden_v1 = model(inputs)
        outputs_v2, hidden_v2 = model(inputs_2)

        # Loss
        r_loss = recon_loss(outputs_v1, gt_input)
        cl_loss = contrastive_loss(outputs_v1.flatten(start_dim=1), outputs_v2.flatten(start_dim=1))
        total_loss = r_loss + cl_loss * r_loss  # Adjust the weight of CL loss

        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()
        epoch_cl_loss += cl_loss.item()
        epoch_recon_loss += r_loss.item()

        print(f"Step {step}, train_loss: {total_loss.item():.4f}")

    # Calculate the average loss
    num_steps = step
    return epoch_loss / num_steps, epoch_cl_loss / num_steps, epoch_recon_loss / num_steps

# Define the validation function
def validate(model, val_loader, recon_loss,device):
    model.eval()
    val_loss = 0
    val_steps = 0
    with torch.no_grad():
        for val_data in val_loader:
            inputs, gt_input = val_data["image"].to(device), val_data["gt_image"].to(device)
            outputs, _ = model(inputs)
            loss = recon_loss(outputs, gt_input)
            val_loss += loss.item()
            val_steps += 1

    return val_loss / val_steps



def self_pretrain(FLARE_dir, model, lr=1e-4, num_epochs= 500):
    # Define device
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # define output dir
    unlabeled_data_dir = os.path.join(FLARE_dir, "Training", "FLARE22_UnlabeledCase")
    validation_data_dir = os.path.join(FLARE_dir, "Tuning")
    output_path = os.path.join(FLARE_dir, "outputs_self")
    os.makedirs(output_path, exist_ok=True)

    # Load data
    unlabeled_loader = get_unlabeled_data_loader(unlabeled_data_dir, batch_size=4, num_workers=4)
    val_loader = get_labeled_data_loader(validation_data_dir, labeled_transforms=self_sup_transforms, batch_size=4, num_workers=4)

    # model and optimizer
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    recon_loss = torch.nn.L1Loss()
    contrastive_loss = ContrastiveLoss(temperature=0.05)

    # record loss
    epoch_loss_values = []
    epoch_cl_loss_values = []
    epoch_recon_loss_values = []
    val_loss_values = []
    best_val_loss = float("inf")

    # Training and validation
    for epoch in range(42, num_epochs):
        print("-" * 10)
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # training
        epoch_loss, epoch_cl_loss, epoch_recon_loss = train(model, unlabeled_loader, optimizer, recon_loss, contrastive_loss, device)
        epoch_loss_values.append(epoch_loss)
        epoch_cl_loss_values.append(epoch_cl_loss)
        epoch_recon_loss_values.append(epoch_recon_loss)
        print(f"Epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # validation
        val_loss = validate(model, val_loader, recon_loss,device)
        val_loss_values.append(val_loss)
        print(f"Epoch {epoch + 1} average validation loss: {val_loss:.4f}")

        # Save the best model weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_path, "best_model.pth"))
            print(f"Saved best model at Epoch {epoch + 1} with validation loss {best_val_loss:.4f}")

        if (epoch +1) % 1 == 0:
            plot_losses(output_path, epoch_loss_values, val_loss_values, epoch_cl_loss_values, epoch_recon_loss_values, epoch)

    # Save the final model
    torch.save(model.state_dict(), os.path.join(output_path, "final_model.pth"))
    print("Training complete. Final model saved.")

# main
if __name__ == "__main__":
    FLARE_dir = "../../../FLARE"
    self_pretrain(FLARE_dir, model)

