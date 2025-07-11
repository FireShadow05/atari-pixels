"""
Controls:

SPACE: Fire
LEFT ARROW: Move Left
RIGHT ARROW: Move Right
. (PERIOD): No Operation (NOOP) -> DEFAULT
ESC or Q: Quit

Optional Parameters:

--temperature: Control the randomness of predictions (lower = more deterministic)

python play_neural_breakout.py --temperature 0.05
The game runs at a lower FPS (15) to make changes more visible. You can adjust this by changing the FPS variable if you want faster gameplay."""

import os
import sys
import torch
import numpy as np
import pygame
from PIL import Image
import time
import argparse
from latent_action_model import load_latent_action_model, ActionStateToLatentMLP, ActionToLatentMLP

# --- Configuration ---
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 840
FPS = 15  # Slower FPS for visible changes

# --- Device selection ---
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def action_to_onehot(action_idx, N_ACTIONS, device):
    onehot = torch.zeros(1, N_ACTIONS, device=device)
    onehot[0, action_idx] = 1.0
    return onehot

def key_combo_to_action(keys):
    up    = keys[pygame.K_UP]
    down  = keys[pygame.K_DOWN]
    left  = keys[pygame.K_LEFT]
    right = keys[pygame.K_RIGHT]
    fire  = keys[pygame.K_SPACE]

    # base direction
    if up  and right: base = 6   # UPRIGHT
    elif up  and left:  base = 7   # UPLEFT
    elif down and right: base = 8   # DOWNRIGHT
    elif down and left:  base = 9   # DOWNLEFT
    elif up:             base = 2   # UP
    elif right:          base = 3   # RIGHT
    elif left:           base = 4   # LEFT
    elif down:           base = 5   # DOWN
    else:                base = 0   # NOOP

    # add FIRE/JUMP
    if fire and base == 0:
        return 1                   # pure FIRE
    elif fire and base in (2, 3, 4, 5, 6, 7, 8, 9):
        return {2:10, 3:11, 4:12, 5:13, 6:14, 7:15, 8:16, 9:17}[base]
    else:
        return base

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', type=float, default=0.01, help='Sampling temperature for latent prediction')
    args = parser.parse_args()
    
    # Initialize pygame
    pygame.init()
    window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Neural Breakout")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 18)
    
    # Set up device and models
    device = get_device()
    print(f"[INFO] Using device: {device}")
    
    # Load world model
    print("[INFO] Loading world model...")
    world_model, _ = load_latent_action_model('checkpoints/best.pt', device)
    world_model.to(device)
    world_model.eval()
    if device.type == 'cuda':
        world_model = torch.compile(world_model)
    
    # Load action-to-latent model
    print("[INFO] Loading action-to-latent model...")
    action_model = ActionStateToLatentMLP(
            action_dim=18,
            latent_dim=80,
            codebook_size=512,      # ← keep in sync
        ).to(device)
    ckpt = torch.load('checkpoints/action_state_to_latent_best.pt', map_location=device)
    # action_model = ActionToLatentMLP().to(device)
    # ckpt = torch.load('checkpoints/latent_action/action_to_latent_best.pt', map_location=device)
    action_model.load_state_dict(ckpt['model_state_dict'], strict = False)
    action_model.eval()
    if device.type == 'cuda':
        action_model = torch.compile(action_model)
    
    # Load initial frame
    print("[INFO] Loading initial frame...")
    init_img = Image.open('/Users/db/Downloads/28.png').convert('RGB')
    init_frame_np = np.array(init_img, dtype=np.float32) / 255.0
    current_frame = torch.from_numpy(init_frame_np).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Set up frame history
    last2_frames = [current_frame.clone(), current_frame.clone()]
    
    # Action mapping
    # action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
    # --- before main() ---------------------------------------------------
    action_names = [
        "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN",
        "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT",
        "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE",
        "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE"
    ]
    N_ACTIONS = len(action_names) 


    key_to_action = {
        pygame.K_PERIOD: 0,  # NOOP
        pygame.K_SPACE: 1,   # FIRE
        pygame.K_RIGHT: 2,   # RIGHT
        pygame.K_LEFT: 3     # LEFT
    }
    
    # Game state
    action_idx = 0  # Default to NOOP
    last_displayed_action = ""  # For display purposes
    score = 0
    step = 0
    
    # Display the key mappings at the start
    # print("\nNeural Breakout Controls:")
    # print("------------------------")
    # print("SPACE - Fire")
    # print("LEFT ARROW - Move Left")
    # print("RIGHT ARROW - Move Right")
    # print(". (PERIOD) - No Operation")
    # print("ESC or Q - Quit\n")
    
    # Main game loop
    running = True
    while running:
        # Default to NOOP (0) each frame
        action_idx = 0
        # print(action_idx)
        
        # Process events
        for event in pygame.event.get():
            # print('F')
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Handle specific quit keys
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                # Set the action based on key press
                elif event.key in key_to_action:
                    action_idx = key_to_action[event.key]
        
        # Check currently pressed keys (for continuous input)
        # keys = pygame.key.get_pressed()
        # if keys[pygame.K_SPACE]:
        #     action_idx = 1  # FIRE
        #     last_displayed_action = "FIRE"
        # elif keys[pygame.K_RIGHT]:
        #     action_idx = 2  # RIGHT
        #     last_displayed_action = "RIGHT"
        # elif keys[pygame.K_LEFT]:
        #     action_idx = 3  # LEFT
        #     last_displayed_action = "LEFT"
        # elif keys[pygame.K_PERIOD]:
        #     action_idx = 0  # NOOP (explicitly)
        #     last_displayed_action = "NOOP"
        keys = pygame.key.get_pressed()
        # print(keys)
        action_idx = key_combo_to_action(keys)          # ← single call
        last_displayed_action = action_names[action_idx]
        
        # Only print non-NOOP actions to console
        if action_idx != 0:
            print(f"Action: {action_names[action_idx]}")
        
        # Generate next frame
        with torch.no_grad():
            # Stack last 2 frames
            stacked_frames = torch.cat([last2_frames[0], last2_frames[1]], dim=1)
            
            # Get action prediction
            onehot = action_to_onehot(action_idx, N_ACTIONS, device)
            logits = action_model(onehot, stacked_frames)
            indices = action_model.sample_latents(logits, temperature=args.temperature)
            
            # Reshape indices and get embeddings
            indices = indices.view(1, 8, 10)
            embeddings = world_model.vq.embeddings
            indices = indices.to(embeddings.weight.device)
            quantized = embeddings(indices)
            quantized = quantized.permute(0, 3, 1, 2).contiguous()
            
            # Generate next frame
            frame_in = current_frame.permute(0, 1, 3, 2)
            quantized = quantized.to(device)
            frame_in = frame_in.to(device)
            next_frame = world_model.decoder(quantized, frame_in)
            next_frame = next_frame.permute(0, 1, 3, 2)
            
            # Update frame history
            last2_frames[0] = last2_frames[1]
            last2_frames[1] = next_frame.clone()
            current_frame = next_frame.clone()
        
        # Convert the frame to a pygame surface for display
        frame_np = current_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
        
        # Create surface from numpy array
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame_np, (1, 0, 2)))
        
        # Scale the surface to fit the window
        scaled_surface = pygame.transform.scale(frame_surface, (WINDOW_WIDTH, WINDOW_HEIGHT - 40))
        
        # Render to the window
        window.fill((0, 0, 0))
        window.blit(scaled_surface, (0, 0))
        
        # Display game info - always show the last action
        step += 1
        info_text = f"Step: {step}"
        if last_displayed_action:
            info_text += f" | Last Action: {last_displayed_action}"
        info_text += f" | Temperature: {args.temperature}"
        
        text_surface = font.render(info_text, True, (255, 255, 255))
        window.blit(text_surface, (10, WINDOW_HEIGHT - 30))
        
        # Update the display
        pygame.display.flip()
        
        # Limit framerate
        clock.tick(FPS)
    
    # Clean up
    pygame.quit()
    print("Game closed.")

if __name__ == "__main__":
    main()