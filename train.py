import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import random


from third_part.face3d.models.arcface_torch.backbones.iresnet import iresnet100

from models import load_DNet, load_network
import torch.nn as nn
import torch.nn.functional as F


class VGG19Features(nn.Module):
    def __init__(self, layer_indices, requires_grad=False):
        super(VGG19Features, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), # 0, 1 (relu1_1)
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), # 2, 3 (relu1_2) -> idx 3
            nn.MaxPool2d(kernel_size=2, stride=2), # 4
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), # 5, 6 (relu2_1)
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True), # 7, 8 (relu2_2) -> idx 8
            nn.MaxPool2d(kernel_size=2, stride=2), # 9
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), # 10, 11(relu3_1)
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), # 12, 13(relu3_2)
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), # 14, 15(relu3_3)
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), # 16, 17(relu3_4) -> idx 17
        )
        self.layer_indices = layer_indices

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        self.eval() # Set to eval mode

    def forward(self, x):
        output_features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layer_indices:
                output_features.append(x)
        return output_features

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 features (L2 norm based on paper)."""
    def __init__(self, layer_indices=[3, 8, 17], requires_grad=False): # Example: relu1_2, relu2_2, relu3_4
        super(PerceptualLoss, self).__init__()
        self.vgg_features = VGG19Features(layer_indices=layer_indices, requires_grad=requires_grad)
        self.criterion = nn.MSELoss() # Use MSELoss for L2 norm ||.||_2

    def forward(self, x, y):
        x_features = self.vgg_features(x)
        y_features = self.vgg_features(y)
        loss = 0.0
        for x_feat, y_feat in zip(x_features, y_features):
             loss += self.criterion(x_feat, y_feat.detach())
        return loss

class GramMatrix(nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())  # compute the gram product
        return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    """Style loss using Gram matrix of VGG19 features (L2 norm based on paper Eq 3)."""
    def __init__(self, layer_indices=[3, 8, 17], requires_grad=False): # Example: relu1_2, relu2_2, relu3_4
        super(StyleLoss, self).__init__()
        self.vgg_features = VGG19Features(layer_indices=layer_indices, requires_grad=requires_grad)
        self.criterion = nn.MSELoss() # Use MSELoss for L2 norm ||.||_2
        self.gram = GramMatrix()

    def forward(self, x, y):
        x_features = self.vgg_features(x)
        y_features = self.vgg_features(y)
        loss = 0.0
        for x_feat, y_feat in zip(x_features, y_features):
            gram_x = self.gram(x_feat)
            gram_y = self.gram(y_feat.detach())
            loss += self.criterion(gram_x, gram_y)
        return loss

class LipSyncLoss(nn.Module):
    """Lip Sync loss using a pre-trained SyncNet."""
    def __init__(self, device):
        super(LipSyncLoss, self).__init__()
        self.criterion = nn.BCELoss() # Binary Cross Entropy for sync probability
        print("Placeholder SyncNet initialized. Replace with actual model.")

    def forward(self, mel, vid_frames):
        loss = torch.tensor(0.0, device=mel.device) # Placeholder value
        return loss

class IdentityLoss(nn.Module):
    """Identity loss using a pre-trained face recognition network (ArcFace - iresnet100)."""
    def __init__(self, device, checkpoint_path="checkpoints/arcface_iresnet100.pth"): # Placeholder path
        super(IdentityLoss, self).__init__()
        self.facenet = iresnet100(fp16=False) # Assuming not using fp16 for training loss
        self.facenet_loaded = False
        try:
            self.facenet.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))) # Load weights
            print(f"Loading ArcFace checkpoint from: {checkpoint_path}")
            self.facenet_loaded = True
        except FileNotFoundError:
             print(f"ERROR: ArcFace checkpoint not found at {checkpoint_path}. Identity loss will be disabled.")
             self.facenet = None # Set to None to indicate failure
             self.criterion = None
             self.face_pool = None
             return # Exit init early
        except Exception as e:
             print(f"ERROR: Failed to load ArcFace checkpoint from {checkpoint_path}: {e}. Identity loss will be disabled.")
             self.facenet = None
             self.criterion = None
             self.face_pool = None
             return

        self.facenet.to(device)
        self.facenet.eval() # Set to evaluation mode
        for param in self.facenet.parameters():
            param.requires_grad = False # Freeze weights

        self.criterion = nn.MSELoss() # Paper uses L2 norm (||.||_2) in Eq 12
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112)) # ArcFace expects 112x112 input

    def _preprocess(self, img):
        if not self.facenet_loaded or self.face_pool is None: return None # Check if init failed
        img = self.face_pool(img) # Resize to 112x112
        img = (img * 2) - 1 # Normalize to [-1, 1]
        return img

    def forward(self, generated_face, ground_truth_face):
        if not self.facenet_loaded or self.facenet is None: # Check if init failed
            return torch.tensor(0.0, device=generated_face.device, requires_grad=False) # Return zero loss if model not loaded

        generated_face_processed = self._preprocess(generated_face)
        ground_truth_face_processed = self._preprocess(ground_truth_face)

        if generated_face_processed is None or ground_truth_face_processed is None:
             return torch.tensor(0.0, device=generated_face.device, requires_grad=False)

        emb_gen = self.facenet(generated_face_processed)
        emb_gt = self.facenet(ground_truth_face_processed)

        loss = self.criterion(emb_gen, emb_gt.detach()) if self.criterion is not None else torch.tensor(0.0, device=emb_gen.device)
        return loss



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, padding=1) # Output a single value (or patch)
        )
        print("Placeholder Discriminator initialized.")

    def forward(self, img):
        return self.model(img)




class VideoDataset(Dataset):
    """
    Dataset class for VideoReTalking training.
    Loads video frames, audio mel-spectrograms, and potentially 3DMM coefficients.
    Needs adaptation based on the specific dataset structure (e.g., LRS2, VoxCeleb).
    """
    def __init__(self, data_root, args, frame_length=5, img_size_low=96, img_size_high=256):
        super().__init__()
        self.data_root = data_root
        self.args = args
        self.frame_length = frame_length # Number of frames per sample (T)
        self.img_size_low = img_size_low
        self.img_size_high = img_size_high
        self.file_list = self._scan_dataset(data_root)
        if not self.file_list:
             print(f"Warning: No data found in {data_root}. Using dummy data.")
             self.file_list = ["dummy_video_path_1", "dummy_video_path_2"] * 10 # Dummy data if scan fails
        print(f"Dataset initialized with {len(self.file_list)} samples.")

    def _scan_dataset(self, data_root):
        """Scans the data root directory to find video/audio pairs."""
        print(f"Scanning {data_root} for dataset files... (Placeholder)")
        return [] # Return empty list for now

    def _load_video_frames(self, video_path):
        """Loads and preprocesses video frames."""
        print(f"Loading video frames from {video_path}... (Placeholder)")
        dummy_frames_low = torch.randn(self.frame_length, 3, self.img_size_low, self.img_size_low)
        dummy_frames_high = torch.randn(self.frame_length, 3, self.img_size_high, self.img_size_high)
        return dummy_frames_low, dummy_frames_high

    def _load_audio_mel(self, audio_path):
        """Loads audio and computes mel-spectrogram."""
        print(f"Loading audio mel from {audio_path}... (Placeholder)")
        dummy_audio_mel = torch.randn(1, 80, 16) # B=1, Mel bins, Time steps (adjust time steps based on frame_length/audio)
        return dummy_audio_mel

    def _load_coeffs(self, coeff_path):
        """Loads 3DMM coefficients."""
        print(f"Loading coeffs from {coeff_path}... (Placeholder)")
        dummy_coeffs = torch.randn(68) # Placeholder shape
        return dummy_coeffs

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        video_identifier = self.file_list[idx] # This might be a path or an ID

        video_path = f"{video_identifier}.mp4" # Example
        audio_path = f"{video_identifier}.wav" # Example
        coeff_path = f"{video_identifier}_coeffs.npy" # Example

        gt_frames_low, gt_frames_high = self._load_video_frames(video_path)
        audio_mel = self._load_audio_mel(audio_path)
        coeffs = self._load_coeffs(coeff_path)

        source_frame_low = gt_frames_low[0]
        source_frame_high = gt_frames_high[0] # Keep high-res source if needed

        sample = {
            'source_frame_low': source_frame_low, # Single source frame (C, H, W)
            'source_frame_high': source_frame_high, # Single source frame (C, H, W)
            'video_frames_low': gt_frames_low, # Sequence of GT frames (T, C, H, W)
            'video_frames_high': gt_frames_high, # Sequence of GT frames (T, C, H, W)
            'audio_mel': audio_mel, # Mel spectrogram (B=1, Mel, Time)
            'coeffs': coeffs # Coefficients (Shape depends on 3DMM)
        }
        return sample

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    train_dataset = VideoDataset(data_root=args.dataset_path, args=args, frame_length=5, img_size_low=96, img_size_high=256)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    DNet = load_DNet(args).to(device)
    LNet, ENet = load_network(args)
    LNet = LNet.to(device)
    ENet = ENet.to(device)

    if args.mode == 'finetune':
        if args.finetune_checkpoint_d and os.path.exists(args.finetune_checkpoint_d):
            print(f"Loading DNet checkpoint: {args.finetune_checkpoint_d}")
            DNet.load_state_dict(torch.load(args.finetune_checkpoint_d, map_location=lambda storage, loc: storage))
        else:
            print("Warning: DNet checkpoint not found or not specified for fine-tuning.")

        if args.finetune_checkpoint_l and os.path.exists(args.finetune_checkpoint_l):
            print(f"Loading LNet checkpoint: {args.finetune_checkpoint_l}")
            LNet.load_state_dict(torch.load(args.finetune_checkpoint_l, map_location=lambda storage, loc: storage))
        else:
            print("Warning: LNet checkpoint not found or not specified for fine-tuning.")

        if args.finetune_checkpoint_e and os.path.exists(args.finetune_checkpoint_e):
            print(f"Loading ENet checkpoint: {args.finetune_checkpoint_e}")
            ENet.load_state_dict(torch.load(args.finetune_checkpoint_e, map_location=lambda storage, loc: storage))
        else:
            print("Warning: ENet checkpoint not found or not specified for fine-tuning.")

    params_dnet = DNet.parameters()
    params_lnet = LNet.parameters()
    params_enet_generator = ENet.parameters() # Assuming ENet is the generator part for adversarial loss


    optimizer_d = optim.Adam(params_dnet, lr=args.lr_dnet, betas=(0.9, 0.999))
    optimizer_l = optim.Adam(params_lnet, lr=args.lr_lnet, betas=(0.9, 0.999))
    optimizer_e_gen = optim.Adam(params_enet_generator, lr=args.lr_enet_gen, betas=(0.9, 0.999))

    ENet_discriminator = Discriminator().to(device) # Instantiate Discriminator
    optimizer_e_disc = optim.Adam(ENet_discriminator.parameters(), lr=args.lr_enet_disc, betas=(0.9, 0.999))

    if args.mode == 'finetune':
        if args.finetune_checkpoint_e_disc and os.path.exists(args.finetune_checkpoint_e_disc):
            print(f"Loading ENet_Disc checkpoint: {args.finetune_checkpoint_e_disc}")
            ENet_discriminator.load_state_dict(torch.load(args.finetune_checkpoint_e_disc, map_location=lambda storage, loc: storage))
        else:
             print("Warning: ENet_Disc checkpoint not found or not specified for fine-tuning.")

    criterion_adversarial = nn.MSELoss() # Assuming LSGAN (Least Squares GAN) for stability

    criterion_l1 = nn.L1Loss()
    criterion_perceptual = PerceptualLoss().to(device)
    criterion_style = StyleLoss().to(device) # For D-Net editing network
    criterion_sync = LipSyncLoss(device).to(device) # For L-Net
    criterion_identity = IdentityLoss(device).to(device) # For E-Net

    lambda_d_c = 1.0       # D-Net Content (Perceptual) Eq.4
    lambda_d_s = 250.0    # D-Net Style (Gram) Eq.4
    lambda_l_l1 = 1.0     # L-Net L1 Eq.9
    lambda_l_p = 1.0      # L-Net Perceptual Eq.9
    lambda_l_sync = 0.3   # L-Net Sync Eq.9
    lambda_e_l1 = 0.2     # E-Net L1 Eq.14
    lambda_e_p = 1.0      # E-Net Perceptual Eq.14
    lambda_e_adv = 100.0  # E-Net Adversarial Eq.14
    lambda_e_id = 0.4     # E-Net Identity Eq.14

    os.makedirs(args.output_dir, exist_ok=True)

    print("Starting training...")
    for epoch in range(args.epochs):
        DNet.train()
        LNet.train()
        ENet.train()

        epoch_loss_e_disc = 0.0 # Add tracker for discriminator loss

        epoch_loss_d = 0.0
        epoch_loss_l = 0.0
        epoch_loss_e = 0.0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for i, batch in enumerate(progress_bar):
            source_frame_low = batch['source_frame_low'].to(device)     # (B, C, H_low, W_low)
            target_frames_low = batch['video_frames_low'].to(device)    # (B, T, C, H_low, W_low)
            target_frames_high = batch['video_frames_high'].to(device)  # (B, T, C, H_high, W_high)
            audio_mels = batch['audio_mel'].to(device)                  # (B, 1, Mel, Time) - Check shape
            coeffs = batch['coeffs'].to(device)                         # (B, Coeff_dim) - Check shape

            optimizer_d.zero_grad()
            target_frame_d_low = target_frames_low[:, 0, ...] # (B, C, H_low, W_low)
            target_coeffs_d = coeffs # Assuming coeffs correspond to the target frame/identity

            generated_frame_d = DNet(source_frame_low, target_coeffs_d) # Output I'_T (B, C, H_low, W_low)

            loss_d_c = criterion_perceptual(generated_frame_d, target_frame_d_low) # Eq. 2
            loss_d_s = criterion_style(generated_frame_d, target_frame_d_low)      # Eq. 3
            loss_d = lambda_d_c * loss_d_c + lambda_d_s * loss_d_s             # Eq. 4

            loss_d.backward()
            optimizer_d.step()
            epoch_loss_d += loss_d.item()

            optimizer_l.zero_grad()
            input_frame_l = generated_frame_d.detach() # Use stabilized frame
            target_sequence_l = target_frames_low # (B, T, C, H_low, W_low)

            generated_frame_l = LNet(input_frame_l, audio_mels) # Output O_LR (B, C, H_low, W_low)
            target_frame_l = target_sequence_l[:, 0, ...] # Target is the corresponding GT frame

            loss_l1 = criterion_l1(generated_frame_l, target_frame_l)                 # Eq. 5
            loss_perceptual_l = criterion_perceptual(generated_frame_l, target_frame_l) # Eq. 6
            loss_sync = criterion_sync(audio_mels, generated_frame_l)                 # Eq. 7 (Placeholder)
            loss_l = lambda_l_l1 * loss_l1 + lambda_l_p * loss_perceptual_l + lambda_l_sync * loss_sync # Eq. 9

            loss_l.backward()
            optimizer_l.step()
            epoch_loss_l += loss_l.item()


            optimizer_e_disc.zero_grad()
            target_frame_e = target_frames_high[:, 0, ...] # Use corresponding high-res GT frame (B, C, H_high, W_high)
            real_output = ENet_discriminator(target_frame_e)
            loss_d_real = criterion_adversarial(real_output, torch.ones_like(real_output)) # Label real as 1

            generated_frame_l_detached = generated_frame_l.detach() # Detach L-Net output
            generated_frame_e = ENet(generated_frame_l_detached) # Output O_HR (B, C, H_high, W_high)
            fake_output = ENet_discriminator(generated_frame_e.detach()) # Detach E-Net output for discriminator
            loss_d_fake = criterion_adversarial(fake_output, torch.zeros_like(fake_output)) # Label fake as 0

            loss_d_e = (loss_d_real + loss_d_fake) / 2 # Total discriminator loss
            loss_d_e.backward() # Calculate gradients for discriminator
            optimizer_e_disc.step() # Update discriminator weights
            epoch_loss_e_disc += loss_d_e.item() # Track discriminator loss

            optimizer_e_gen.zero_grad()

            fake_output_for_gen = ENet_discriminator(generated_frame_e) # Pass E-Net output through updated discriminator
            loss_adv_gen = criterion_adversarial(fake_output_for_gen, torch.ones_like(fake_output_for_gen)) # Generator wants discriminator to predict 1

            loss_e_l1 = criterion_l1(generated_frame_e, target_frame_e)             # Eq. 10
            loss_perceptual_e = criterion_perceptual(generated_frame_e, target_frame_e) # Eq. 11
            loss_identity = criterion_identity(generated_frame_e, target_frame_e)     # Eq. 12 (Placeholder)

            loss_e = (lambda_e_l1 * loss_e_l1 +
                      lambda_e_p * loss_perceptual_e +
                      lambda_e_id * loss_identity +
                      lambda_e_adv * loss_adv_gen) # Eq. 14

            loss_e.backward() # Calculate gradients for generator
            optimizer_e_gen.step() # Update generator (E-Net) weights
            epoch_loss_e += loss_e.item() # Track combined generator loss


            if (i + 1) % args.log_interval == 0:
                progress_bar.set_postfix({
                    'Loss_D': loss_d.item(),
                    'Loss_L': loss_l.item(),
                    'Loss_E_Gen': loss_e.item(), # Combined ENet Generator loss
                    'Loss_E_Disc': loss_d_e.item() # ENet Discriminator loss
                })

        avg_epoch_loss_d = epoch_loss_d / len(train_dataloader)
        avg_epoch_loss_l = epoch_loss_l / len(train_dataloader)
        avg_epoch_loss_e_gen = epoch_loss_e / len(train_dataloader)
        avg_epoch_loss_e_disc = epoch_loss_e_disc / len(train_dataloader)
        print(f"Epoch {epoch+1} finished. Avg Losses: D={avg_epoch_loss_d:.4f}, L={avg_epoch_loss_l:.4f}, E_Gen={avg_epoch_loss_e_gen:.4f}, E_Disc={avg_epoch_loss_e_disc:.4f}")

        if (epoch + 1) % args.save_interval == 0:
            dnet_path = os.path.join(args.output_dir, f'DNet_epoch_{epoch+1}.pt')
            lnet_path = os.path.join(args.output_dir, f'LNet_epoch_{epoch+1}.pth')
            enet_gen_path = os.path.join(args.output_dir, f'ENet_Gen_epoch_{epoch+1}.pth') # ENet Generator
            enet_disc_path = os.path.join(args.output_dir, f'ENet_Disc_epoch_{epoch+1}.pth') # ENet Discriminator

            torch.save(DNet.state_dict(), dnet_path)
            torch.save(LNet.state_dict(), lnet_path)
            torch.save(ENet.state_dict(), enet_gen_path) # Save ENet Generator state
            torch.save(ENet_discriminator.state_dict(), enet_disc_path) # Save ENet Discriminator state
            print(f"Checkpoints saved for epoch {epoch+1}")

    



    











    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VideoReTalking Models")

    parser.add_argument('--dataset_path', type=str, required=True, help='Root directory of the training dataset')
    parser.add_argument('--output_dir', type=str, default='./training_checkpoints', help='Directory to save checkpoints')

    parser.add_argument('--mode', type=str, choices=['finetune', 'scratch'], default='scratch', help='Training mode: finetune or train from scratch')

    parser.add_argument('--finetune_checkpoint_d', type=str, default=None, help='Path to DNet checkpoint for fine-tuning')
    parser.add_argument('--finetune_checkpoint_l', type=str, default=None, help='Path to LNet checkpoint for fine-tuning')
    parser.add_argument('--finetune_checkpoint_e', type=str, default=None, help='Path to ENet checkpoint for fine-tuning')

    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--lr_dnet', type=float, default=1e-4, help='Learning rate for D-Net (Appendix C.1.3)')
    parser.add_argument('--lr_lnet', type=float, default=1e-4, help='Learning rate for L-Net (Appendix C.2.3)')
    parser.add_argument('--lr_enet_gen', type=float, default=1e-5, help='Learning rate for E-Net Generator (Appendix C.3.3)')
    parser.add_argument('--lr_enet_disc', type=float, default=1e-5, help='Learning rate for E-Net Discriminator (if used, assume same as generator)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    parser.add_argument('--log_interval', type=int, default=10, help='Log training status every N batches')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint every N epochs')


    args = parser.parse_args()


    main(args)
